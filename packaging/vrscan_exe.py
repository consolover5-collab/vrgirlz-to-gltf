"""PyInstaller's entry script (`vrscan.spec`) — all of it is `vrscan.frozen`."""

from vrscan.frozen import main

raise SystemExit(main())
