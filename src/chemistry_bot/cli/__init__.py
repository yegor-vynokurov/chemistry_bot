"""Command-line entry points for chemistry_bot.

Keep this package initializer side-effect free. In particular, do not import
`rag_bot` here: `python -m chemistry_bot.cli.rag_bot` first imports the
`chemistry_bot.cli` package, and importing the target module from this file
preloads it before runpy executes it, producing:

    RuntimeWarning: 'chemistry_bot.cli.rag_bot' found in sys.modules ...

Import CLI classes directly from their modules instead, for example:

    from chemistry_bot.cli.rag_bot import CliBot
"""

__all__: list[str] = []
