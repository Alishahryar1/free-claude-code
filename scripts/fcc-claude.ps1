param([Parameter(ValueFromRemainingArguments=$true)]$Args)
& python -c "from free_claude_code.cli.launchers.claude import launch; launch()" @Args
