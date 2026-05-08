# Mac Base / Pro 16 GB Setup Guide

Este documento descreve as adaptacoes necessarias para rodar
`claude-code-local` em MacBooks Apple Silicon de entrada/Pro com **16 GB
de memoria unificada** - hardware abaixo do alvo original do projeto
(M Max / Ultra com 64-128 GB).

A documentacao oficial do repositorio assume Mac Max/Ultra. Em hardware
mais modesto, seis problemas reproduziveis aparecem em sequencia:

1. Tela de selecao de login do Claude Code mesmo com `ANTHROPIC_API_KEY` setado
2. Vazamento de tokens internos do modelo (`<|im_end|>`, `<|endoftext|>`, ...) na resposta
3. Respostas vazias ("(No output)") em todas as mensagens
4. Tool-calls do Qwen 2.5 nao reconhecidos pelo parser do servidor
5. Claude Code chama `api.anthropic.com` no startup mesmo com `ANTHROPIC_BASE_URL` setado (vazamento de "100% offline")
6. **Tools nunca executam:** Claude Code descarta toda resposta com `tool_use` porque o servidor responde JSON unico em vez de `text/event-stream`

Este guia explica a causa de cada um e a correcao aplicada nesta branch.

---

## 1. Bug do macOS keychain - tela de login persistente

### Sintoma

Apos rodar o launcher, voce ve a tela de selecao do Claude Code:

```
Select login method:
  1. Claude account with subscription
  2. Anthropic Console account
  3. 3rd-party platform
```

Mesmo com `ANTHROPIC_API_KEY=sk-local` exportado no ambiente.

### Causa

Bug conhecido do Claude Code 2.1.x no macOS - issues
[#25069](https://github.com/anthropics/claude-code/issues/25069) e
[#27900](https://github.com/anthropics/claude-code/issues/27900) no
repositorio oficial. A logica de verificacao do keychain do macOS roda
**antes** da leitura da variavel de ambiente, e quando o keychain esta
vazio o CLI cai no fluxo OAuth em vez de usar a env var.

A flag `--bare` que o repositorio original passa nos launchers nao
existe nas versoes recentes do CLI.

### Correcao

Combinacao de tres ajustes:

1. Setar `hasCompletedOnboarding: true` em `~/.claude.json`:
   ```bash
   python3 -c "
   import json, pathlib
   p = pathlib.Path.home() / '.claude.json'
   d = json.loads(p.read_text()) if p.exists() else {}
   d['hasCompletedOnboarding'] = True
   p.write_text(json.dumps(d, indent=2))
   "
   ```

2. Exportar `ANTHROPIC_AUTH_TOKEN` junto com `ANTHROPIC_API_KEY` (a
   presenca do auth token destrava o caminho de API key no modo
   interativo).

3. Setar `DISABLE_LOGIN_COMMAND=1` para esconder o comando `/login`
   dentro da sessao interativa.

Os launchers `Claude Chat.command` e `Claude Agentico.command` aplicam
os tres automaticamente.

---

## 2. Vazamento de tokens internos do modelo

### Sintoma

Em vez da resposta esperada, voce ve:

```
> me fale sobre typescript
<|endoftext|><|im_start|>user
<system-
```

Os tokens especiais do tokenizer aparecem no texto da resposta.

### Causa

O `clean_response` em `proxy/server.py` so tinha logica para os stop
markers do **Gemma 4** (`<turn|>`, `<|turn>`). Quando o servidor roda
qualquer outro modelo - Qwen, Llama, modelos ChatML em geral - os
markers `<|im_end|>`, `<|im_start|>`, `<|endoftext|>`, `<|eot_id|>` nao
eram tratados. O modelo sinalizava fim de turno corretamente, mas o
servidor nao truncava no marker e o texto vazava.

### Correcao

Patch em `proxy/server.py:132`:

```python
# Antes
for stop_marker in ['<turn|>', '<|turn>']:

# Depois
for stop_marker in ['<turn|>', '<|turn>', '<|im_end|>', '<|endoftext|>',
                    '<|im_start|>', '<|end_of_text|>', '<|eot_id|>']:
```

Truncamento de saida no primeiro marker encontrado, agora cobrindo
ChatML (Qwen, Mistral, varios modelos), Llama 3.x e Gemma.

---

## 3. Extended thinking quebra modelos pequenos

### Sintoma

Toda mensagem retorna "(No output)" no Claude Code, mesmo sem tools
e mesmo apos o fix de stop markers.

No log do servidor MLX:

```
[06:14:25] POST /v1/messages tools=25
[06:14:25]   Generated: 27 tokens - "O usuario esta testando o sistema..."
[06:14:25] POST /v1/messages tools=25
[06:14:25]   Generated: 6 tokens - "(No output)"
```

### Causa

O Claude Code 2.1 introduziu **extended thinking**: para cada turno,
faz duas chamadas ao modelo - a primeira pede `thinking` (cadeia de
raciocinio interna), a segunda pede `text` (resposta final).

Modelos pequenos / quantizados gastam o orcamento todo na primeira
chamada (raciocinando) e quando recebem a segunda nao tem mais o que
gerar - emitem so `<|im_end|>` direto e o texto sai vazio.

### Correcao

Forcar `--effort low` no Claude Code, que desativa o extended thinking
e faz cada turno virar uma unica chamada:

```bash
claude --model claude-sonnet-4-6 --effort low ...
```

Aplicado por padrao nos dois launchers desta branch.

---

## 4. Tool-calls do Qwen 2.5 nao reconhecidos

### Sintoma

Modo agentico nao executa nada. O modelo gera o pedido de tool, mas o
Claude Code nao executa - aparece como texto puro:

```
> use Bash to print pwd
<tools>
{"name": "Bash", "arguments": {"command": "pwd"}}
</tools>
```

### Causa

O `parse_tool_calls` em `proxy/server.py` cobria `<tool_call>...</tool_call>`
(formato Qwen 3.5 e variantes ChatML modernas) e `<|tool_call>...<tool_call|>`
(Gemma 4 nativo), mas **nao** cobria `<tools>...</tools>` - o formato que o
Qwen 2.5 Coder 14B emite naturalmente.

Sem o parser reconhecer a tag, o servidor devolvia o conteudo como bloco
de texto e o Claude Code nao tinha como executar.

### Correcao

Adicionado **Format 3.5** ao `parse_tool_calls` para reconhecer o
wrapper `<tools>...</tools>`, com suporte a payload unico (objeto JSON)
ou multiplos (array de objetos):

```python
pattern_qwen25 = r'<tools>\s*(.*?)\s*</tools>'
for match in re.finditer(pattern_qwen25, text, re.DOTALL):
    content = match.group(1).strip()
    call_data = json.loads(content)
    # aceita lista [{"name":..,"arguments":..}, ...] ou um objeto unico
```

Validado com `Bash({"command":"pwd"})` retornando `stop_reason: tool_use`.

---

## 5. Vazamento "100% offline" - Claude Code chama api.anthropic.com no startup

Esta foi a descoberta mais grave durante a investigacao desta branch e
nao tem nenhuma mencao no README do upstream.

### Sintoma

Apos aplicar todos os fixes anteriores, ao rodar:

```bash
ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-local \
claude --print --tools "" "Hi"
```

O processo trava por minutos sem qualquer chamada chegar ao servidor
MLX local. Inspecionando com `lsof -p $PID`:

```
claude  25331  TCP mac:63057->160.79.104.10:https (ESTABLISHED)
```

O IP `160.79.104.10` resolve para **Anthropic, PBC** (whois confirma) e
e o IP de `api.anthropic.com`. Ou seja, **mesmo com `ANTHROPIC_BASE_URL`
apontando para localhost:4000, o Claude Code abre conexao HTTPS para o
servidor real da Anthropic na inicializacao.**

Isso e um vazamento serio para qualquer pessoa que pensa estar rodando
"100% offline" - codigo, prompts e telemetria estao saindo da maquina
mesmo com o setup local funcionando.

### Causa

O Claude Code 2.1.x dispara, na inicializacao, uma serie de chamadas
"nao essenciais" antes de processar o prompt do usuario:

- Telemetria (`tengu_*` events)
- Feature flags via Statsig
- Auto-install do marketplace oficial de plugins
- Verificacao de auto-updater
- Background tasks scheduling

Todas essas chamadas vao para `api.anthropic.com` e `events.statsig.com`
diretamente, sem passar pelo `ANTHROPIC_BASE_URL`. Isso esta confirmado
nas strings do binario:

```
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC
CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL
DISABLE_AUTOUPDATER
CLAUDE_CODE_DISABLE_BACKGROUND_TASKS
```

### Correcao

Exportar as quatro variaveis de ambiente nos launchers:

```bash
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
DISABLE_AUTOUPDATER=1
CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL=1
CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1
```

Com `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` confirmamos via
`lsof -p $CLAUDE_PID` que **nenhuma** conexao TCP sai para o IP
160.79.104.10 - o claude conecta exclusivamente em `localhost:4000`
(o servidor MLX desta branch).

### Validacao

Apos os fixes, rodando:

```bash
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-local \
ANTHROPIC_AUTH_TOKEN=sk-local \
claude --print --tools "" --effort low "Hi" </dev/null
```

Saida obtida:

```
Hello! How can I assist you today?
```

Servidor MLX (`/tmp/mlx-server.log`):

```
[21:31:07]   Generated: 10 tokens in 63.4s (0.2 tok/s)
[21:31:07]   ← OK (10 tok) Hello! How can I assist you today?...
```

**Nenhuma** chamada saiu da maquina. Verificado com lsof.

> Observacao adicional: em modo `--print`, e necessario fechar
> o stdin com `</dev/null`, caso contrario o claude trava esperando
> input mesmo com o prompt passado como argumento.

---

## 6. Bug critico do modo agentico - Claude Code descarta tool_use sem SSE

Esse foi o problema mais dificil de diagnosticar e o que finalmente
destravou o modo agentico em 16 GB.

### Sintoma

Em sessoes com tools:

```bash
claude "Use Bash para criar hello.txt com 'oi'" --print --dangerously-skip-permissions
```

Saida do claude: `(No output)`. Nenhum arquivo criado.

No log do servidor MLX o que se ve eh:

```
POST /v1/messages tools=22
  ← OK (68 tok) [tool_use: Bash]
POST /v1/messages tools=22         ← Claude Code repete a MESMA request
  DEBUG msg trail: [user] ['text','text','text']   ← sem assistant nem tool_result!
  ← OK (15 tok) (No output)
```

Ou seja: o servidor gera o `tool_use` Bash corretamente, com `id`,
`name`, `input.command` e `stop_reason: "tool_use"` validos. **Mas o
Claude Code descarta a resposta inteira** e refaz a request original
sem incluir o `assistant.tool_use` nem o `user.tool_result`. Por isso
o num_turns reportado pelo CLI eh `1`: do ponto de vista dele, a
primeira chamada nem aconteceu.

### Causa

Claude Code 2.1 envia `stream: true` em **toda** request que tem tools
no body. Quando recebe `Content-Type: application/json` em vez de
`text/event-stream`, ele descarta silenciosamente, retenta uma vez
sem o tool_use no historico, e exibe `(No output)`.

O `proxy/server.py` original ignorava o campo `stream` e sempre
respondia com JSON unico via `send_json`. Para chat puro isso
funciona porque a resposta cabe em um delta. Para tool_use, nao.

### Correcao

Implementar a sequencia de eventos SSE da Messages API da Anthropic
no servidor (`send_anthropic_stream`):

```
event: message_start         <- com id/model/role
event: content_block_start   <- por bloco
event: content_block_delta   <- text_delta para texto, input_json_delta para tools
event: content_block_stop
event: message_delta         <- com stop_reason
event: message_stop
```

E rotear:

```python
if body.get("stream"):
    send_anthropic_stream(self, result)
else:
    send_json(self, 200, result)
```

Detalhe critico: o header `Connection` precisa ser `close` (nao
`keep-alive`). Como o servidor gera a resposta inteira antes de
"streamar" (replay de eventos), nao tem mais nada para mandar. Com
`keep-alive` o cliente fica esperando indefinidamente.

### Validacao

Apos a fix, sessao agentica completa funciona:

```bash
ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-local \
ANTHROPIC_AUTH_TOKEN=sk-local \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
DISABLE_AUTOUPDATER=1 \
claude "Use Bash to run: echo 'oi do qwen' > hello.txt && cat hello.txt" \
  --print --effort low --dangerously-skip-permissions </dev/null
```

Saida do claude:

```
The file `hello.txt` has been created with the content "oi do qwen".
```

`hello.txt` criado no disco com o conteudo correto. Trail observado
no servidor:

```
POST 1: [user] ['text','text','text']                                -> tool_use Bash
POST 2: [user] ['text','text','text'] | [assistant] ['tool_use'] | [user] ['tool_result']
        -> "The file hello.txt has been created..."
```

Multi-tool em sequencia (`Write` + `Read` no mesmo response) tambem
funcionou: o Claude Code executou os dois e enviou os dois
`tool_result` de volta na proxima request.

---

## Modelo recomendado para 16 GB

A tabela oficial do repo nao cobre 16 GB - so 8-16 GB ("modelos pequenos
4B") e 18-36 GB ("Gemma 31B apertado"). Na pratica:

| Modelo | Disco | RAM em uso | Velocidade | Tool-calls | Recomendacao |
|---|---|---|---|---|---|
| Qwen 3.5 4B 4-bit | 2.5 GB | 3-5 GB | 25-40 tok/s | Falha frequente | Apenas modo Chat |
| Llama 3.1 8B 4-bit | 5 GB | 5-7 GB | 15-25 tok/s | Mediano | Compromisso |
| **Qwen 2.5 Coder 14B 4-bit** | **7.8 GB** | **9-11 GB** | **10-15 tok/s** | **OK (Format 3.5)** | **Padrao desta branch** |
| Gemma 4 31B Abliterated 4-bit | 18 GB | 18+ GB | OOM em 16 GB | nao testavel | Inviavel em 16 GB |

Os launchers desta branch usam **Qwen 2.5 Coder 14B 4-bit MLX** por
padrao. Razoes:

- 7.8 GB de pesos cabem em 16 GB de memoria unificada **sem swap**
- O parser `<tools>` (Format 3.5) ja foi adicionado em `proxy/server.py`
- Tool-calls validados com `Bash`/`Read`/`Glob`
- Boa qualidade em codigo e em PT-BR

O Gemma 4 31B foi testado primeiro (eh o padrao do upstream) mas crasha
com `kIOGPUCommandBufferCallbackErrorOutOfMemory` na primeira inferencia
em 16 GB - os 18 GB de pesos simplesmente nao cabem com macOS + apps
abertos consumindo RAM.

Para usar outro modelo, exporte `MLX_MODEL` antes de chamar o launcher:

```bash
MLX_MODEL=mlx-community/Llama-3.1-8B-Instruct-4bit ./launchers/Claude\ Chat.command
```

---

## Limites realistas em 16 GB

O que funciona bem:

- Conversa, perguntas de codigo, geracao de snippets isolados
- Explicacao de conceitos, debugging de stack traces colados
- Codigo sensivel (NDA / juridico / saude) que nao pode sair da maquina
- Trabalho offline (aviao, redes restritas)

O que nao funciona bem:

- Refatoracoes multi-arquivo com Edit/Write em loop
- Sessoes agenticas longas (o KV cache enche e o Mac swappa)
- Tarefas que exigem raciocinio nivel Claude Sonnet/Opus
- Conhecimento factual sobre Brasil / cultura PT-BR (modelos pequenos
  alucinam pesado nesse dominio)

Para essas situacoes, usar o Claude na nuvem (`claude` sem
`ANTHROPIC_BASE_URL`) continua sendo a opcao certa. O setup local eh
complementar, nao substituto, em hardware modesto.
