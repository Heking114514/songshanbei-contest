"""Use platform pip without probing denied OS marketing metadata."""
from pip._vendor import distro

distro.name = lambda *args, **kwargs: ""
distro.version = lambda *args, **kwargs: ""
distro.codename = lambda *args, **kwargs: ""

from pip._internal.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
