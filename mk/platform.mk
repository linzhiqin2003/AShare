# 研究平台常驻服务（macOS launchd）。PLATFORM_SUBNET 是对方访问用的 VPN 网段。
PLATFORM_PORT ?= 8765
PLATFORM_SUBNET ?= 10.77.0.0/24
PLATFORM_LABEL := com.ashare.platform
PLATFORM_PLIST := $(HOME)/Library/LaunchAgents/$(PLATFORM_LABEL).plist
PLATFORM_DOMAIN = gui/$(shell id -u)

.PHONY: platform-service platform-restart platform-stop platform-status

# 安装或更新服务定义并（重新）启动
platform-service:
	@test -f src/ashare/platform/static/index.html || { echo "先运行 make web"; exit 1; }
	@mkdir -p data/platform/logs
	@sed -e 's|{{PYTHON}}|$(CURDIR)/.venv/bin/python|g' -e 's|{{REPO}}|$(CURDIR)|g' \
	     -e 's|{{PORT}}|$(PLATFORM_PORT)|g' -e 's|{{SUBNET}}|$(PLATFORM_SUBNET)|g' \
	     deploy/platform/launchd/$(PLATFORM_LABEL).plist.in > $(PLATFORM_PLIST).tmp
	@if grep -q '{{' $(PLATFORM_PLIST).tmp; then echo "模板占位符没替换完"; exit 1; fi
	@plutil -lint -s $(PLATFORM_PLIST).tmp
	@mv $(PLATFORM_PLIST).tmp $(PLATFORM_PLIST)
	-@launchctl bootout $(PLATFORM_DOMAIN)/$(PLATFORM_LABEL) 2>/dev/null
	@launchctl bootstrap $(PLATFORM_DOMAIN) $(PLATFORM_PLIST)
	@echo "已启动：http://127.0.0.1:$(PLATFORM_PORT)  日志在 data/platform/logs/"

# 改了后端代码或 make web 之后用：只重启进程
platform-restart:
	launchctl kickstart -k $(PLATFORM_DOMAIN)/$(PLATFORM_LABEL)

platform-stop:
	launchctl bootout $(PLATFORM_DOMAIN)/$(PLATFORM_LABEL)

platform-status:
	@launchctl print $(PLATFORM_DOMAIN)/$(PLATFORM_LABEL) | grep -E '^\s*(state|pid|last exit code)' || true
	@tail -n 5 data/platform/logs/platform.out.log 2>/dev/null || true
