# Playwright MCP 安装与配置（pi coding agent）

## 1. 原理
`@playwright/mcp` 是官方 Playwright MCP 服务器，走 **stdio** 传输，由 pi 启动时在后台连接。它把浏览器自动化能力（navigate / snapshot / evaluate / click 等）暴露成 MCP 工具。**无需单独安装服务**——`npx -y` 按需拉取包。

## 2. 前置条件
- **Node.js**（提供 `npx`）
- **浏览器**：二选一
  - 让 MCP 自动下载：`--browser chromium`
  - 复用本机已装 Chrome：`--executable-path "C:\Program Files\Google\Chrome\Application\chrome.exe"`（推荐，省一次下载）

## 3. 配置（两种等价方式）

**方式 A：命令行**
```bash
pi mcp add playwright -- npx -y @playwright/mcp@latest \
  --browser chromium \
  --executable-path "C:\Program Files\Google\Chrome\Application\chrome.exe"
```

**方式 B：直接编辑配置文件**
- 用户级：`~/.pi/agent/mcp.json`（即 `C:\Users\seama\.pi\agent\mcp.json`）
- 项目级：`.pi/mcp.json`（需项目信任后生效，且同名条目覆盖用户级）

```json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": [
        "-y",
        "@playwright/mcp@latest",
        "--browser", "chromium",
        "--executable-path", "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
      ]
    }
  }
}
```

要点：`command` 是单个可执行文件，`args` 是参数数组（**不是 shell 命令串**）；Windows 路径里的 `\` 要写成 `\\`。

## 4. 常用参数
| 参数 | 作用 |
|---|---|
| `--browser <chromium/chrome/firefox/webkit/msedge>` | 选浏览器 |
| `--executable-path <path>` | 指定已装浏览器，跳过下载 |
| `--headless` | 无头模式 |
| `--viewport-size <w,h>` | 默认视口 |
| `--user-agent <ua>` | 自定义 UA |
| `--isolated` | 每次调用用全新隔离上下文（不共享 cookie/历史） |
| `--allowed-origins <origins>` | 白名单限制可访问站点（安全） |

## 5. 验证与诊断
```bash
pi mcp list        # 连接每个已启用服务器，打印状态/工具数/错误；有错时退出码为 1
```
- 会话内 `/mcp` 查看连接状态、工具、暴露级别；改完配置后 `/reload`。
- 若 `/mcp` 中 playwright 显示为 `stopped`，用 `/mcp:start` 启动该 MCP 服务器。
- 日志：`~/.pi/agent/mcp.log`（超过 5MB 轮转为 `mcp.log.1`）。

## 6. 工具命名与暴露（exposure）
- 工具注册名形如 **`mcp__playwright__browser_navigate`**（即 `mcp_playwright_browser_*`）。
- 默认 `exposure` 为 `codemode`；想直接暴露给模型可在配置里加 `"exposure": "direct"`：
```json
{ "playwright": { "command": "npx", "args": [], "exposure": "direct" } }
```

## 7. 安全提示
- 建议用 `--allowed-origins` 限制可访问域名，避免浏览器被模型驱动去访问任意站点。
- MCP 调用走 pi 的工具管线（权限 gate 对 MCP 同样生效），`browser_close` 等破坏性工具会按权限策略处理。
