# Starseed

Se gettingstarted.md för en guide till git 


Om ni tröttnat på att ta bort output
```conf
[filter "strip-notebook-output"]
	clean = "jq '.cells |= map(if .\"cell_type\" == \"code\" then .outputs = [] | .execution_count = null else . end | .metadata = {}) | .metadata = {}'"
```