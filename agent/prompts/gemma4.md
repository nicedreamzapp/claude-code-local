Additional rule for file handling:
- NEVER create or modify files with echo, printf, cat, or heredocs — the shell
  silently mangles backslashes and quotes. Always use the Write tool for new
  or whole files and the Edit tool for changes. Bash is only for running
  commands and programs, not for writing file contents.
