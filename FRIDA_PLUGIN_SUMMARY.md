# IDA Pro Frida Hook 代码生成器 - 最终版本

## 🎯 功能概述

本插件为 IDA Pro 增加了 Frida Hook 代码生成功能，能够为当前选中的函数自动生成完整的 Frida Hook JavaScript 代码。

## ✨ 主要特性

### 1. 版本兼容性
- ✅ **Frida <17.x**: 使用传统 JavaScript 语法和 `Module.findBaseAddress()` API
- ✅ **Frida ≥17.x**: 使用现代 ES6+ 语法和 `Process.getModuleByName().base` API

### 2. 用户界面
- 🔘 **版本选择**: 直观的按钮弹窗（`>=17` / `<17` / `Cancel`）
- 💾 **文件保护**: 覆盖确认弹窗（`Overwrite` / `Cancel`）

### 3. 代码风格
- 📝 使用真实参数名（如 `this.input_str`, `this.len`）
- 🎯 分割符格式：`------------------sub_函数名-------------`
- 📍 地址格式：`base_addr.add(0xAFB48)` + 注释
- 🔍 调用栈信息：完整的 backtrace 输出

### 4. 输出格式
```javascript
// Frida ≥17.x 示例
function hook_sub_AFB48() {
    const base_addr = Process.getModuleByName("your_module.so").base;
    const sub_AFB48_addr = base_addr.add(0x0000AFB48);  //注意，该地址就是函数的地址
    Interceptor.attach(sub_AFB48_addr, {
        onEnter(args) {
            this.input_str = args[0];
            this.len = args[1];
            this.out = args[2];
            console.log(`\n------------------sub_sub_AFB48-------------\n`,
                `[+] Entering sub_AFB48\n`,
                `        input_str:\n${this.input_str}\n`,
                `        len:\n${this.len.toInt32()}\n`,
                `        out:\n${this.out.toInt32()}\n`,
                `setValue called from:\n` + Thread.backtrace(this.context, Backtracer.ACCURATE).map(DebugSymbol.fromAddress).join('\n') + '\r\n'
            );
        },
        onLeave(retval) {
            console.log(`[+] Leaving sub_AFB48, return: ${retval}`);
        }
    })
}

function main() {
    hook_sub_AFB48();
}

setImmediate(main);
```

## 🔧 使用方法

1. **启动**: 在 IDA Pro 中定位到目标函数
2. **生成**: 选择菜单 `Tools → MCP → Generate Frida Hook (Current Function)`
3. **选择版本**: 点击 `>=17` 或 `<17` 按钮
4. **确认覆盖**: 如果文件存在，选择是否覆盖
5. **完成**: 获得完整的 Frida Hook 代码文件

## 📁 核心文件

- `src/ida_pro_mcp/mcp-plugin.py` - 主要实现文件
- `FRIDA_GENERATOR_README.md` - 详细功能文档
- `1.js` - 用户代码风格参考

## 📋 技术规范

### API 差异
- **Frida <17.x**: `Module.findBaseAddress("module.so")`
- **Frida ≥17.x**: `Process.getModuleByName("module.so").base`

### 代码规范
- 参数命名：使用真实参数名，不添加注释
- 输出格式：每个参数单独一行
- 调用栈：包含 `setValue called from:` 前缀
- 函数包装：每个 Hook 包装在独立函数中

## 🎉 完成功能

- ✅ 删除了生成所有函数的功能（专注单个函数）
- ✅ 删除了 MCP 工具接口（仅保留 IDA Pro 插件）
- ✅ 按钮式版本选择和文件覆盖确认
- ✅ 完全模仿用户代码风格
- ✅ 支持 Frida 版本兼容性
- ✅ 干净简洁的参数赋值（无多余注释）

插件现在完全专注于为 IDA Pro 提供高质量的 Frida Hook 代码生成功能！