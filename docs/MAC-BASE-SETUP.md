# Mac Base / Pro 16 GB Setup Guide

Este documento descreve as adaptacoes necessarias para rodar
`claude-code-local` em MacBooks Apple Silicon de entrada/Pro com **16 GB
de memoria unificada** - hardware abaixo do alvo original do projeto
(M Max / Ultra com 64-128 GB).

A documentacao oficial do repositorio assume Mac Max/Ultra. Em hardware
mais modesto, tres problemas reproduziveis aparecem em sequencia:

1. Tela de selecao de login do Claude Code mesmo com `ANTHROPIC_API_KEY` setado
2. Vazamento de tokens internos do modelo (`<|im_end|>`, `<|endoftext|>`, ...) na resposta
3. Respostas vazias ("(No output)") em todas as mensagens

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

## Modelo recomendado para 16 GB

A tabela oficial do repo nao cobre 16 GB - so 8-16 GB ("modelos pequenos
4B") e 18-36 GB ("Gemma 31B apertado"). Na pratica:

| Modelo | Disco | RAM em uso | Velocidade | Tool-calls | Recomendacao |
|---|---|---|---|---|---|
| Qwen 3.5 4B 4-bit | 2.5 GB | 3-5 GB | 25-40 tok/s | Falha frequente | Apenas modo Chat |
| Llama 3.1 8B 4-bit | 5 GB | 5-7 GB | 15-25 tok/s | Mediano | Compromisso |
| Qwen 2.5 Coder 14B 4-bit | 8 GB | 9-11 GB | 10-15 tok/s | OK | Bom para codigo |
| Gemma 4 31B Abliterated 4-bit | 18 GB | 18 GB | 5-8 tok/s | OK (testado pelo upstream) | Tool-calls confiaveis, vai swappar |

Os launchers desta branch usam **Gemma 4 31B** por padrao porque eh o
modelo cujo formato de tool-call ja eh nativo no `proxy/server.py` do
upstream e cuja confiabilidade em tool-calls foi validada pelo autor
original do repo. Em 16 GB de RAM ele swappa cerca de 2 GB para o SSD,
ficando lento mas funcional.

Para usar outro modelo, exporte `MLX_MODEL` antes de chamar o launcher:

```bash
MLX_MODEL=mlx-community/Qwen2.5-Coder-14B-Instruct-4bit ./launchers/Claude\ Chat.command
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
