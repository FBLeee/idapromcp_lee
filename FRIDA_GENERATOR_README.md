# Frida 代码生成器功能（TODO）

## 功能概述

我们成功在 IDA Pro MCP 插件中添加了 Frida 代码生成功能，该功能既可以作为 IDA Pro 插件菜单使用，也可以通过 MCP 工具接口调用。**重要特性：支持 Frida 版本区分，针对 17.x 前后版本的 API 差异生成不同的代码。**

## 🆕 Frida 版本支持

### 支持的版本
- **Frida <17.x**: 传统版本（16.x 及之前）
- **Frida >=17.x**: 现代版本（17.0 及之后）

### 版本差异对比

| 特性         | Frida <17.x     | Frida >=17.x           |
| ------------ | --------------- | ---------------------- |
| 变量声明     | `var`           | `const`/`let`          |
| 函数语法     | `function()`    | 箭头函数支持           |
| 字符串格式化 | 字符串拼接 `+`  | 模板字面量 `` `${}` `` |
| 内存API      | 基础 Memory API | ArrayBuffer.wrap()     |
| 错误处理     | 基础 try-catch  | 增强的错误处理         |
| 性能         | 标准性能        | 优化的内存访问         |

## 功能特性

### 1. 插件菜单模式

在 IDA Pro 中，通过 `Edit/MCP` 菜单可以访问以下功能：

- **Generate Frida Hook (Current Function)** (快捷键: `Ctrl-Alt-F`)
  - 为当前选中的函数生成 Frida Hook 代码
  - 自动保存到 `frida_hooks.js` 文件
  - 默认使用 Frida 17.x 语法

- **Generate Frida Hooks (All Functions)** (快捷键: `Ctrl-Alt-Shift-F`)
  - 为所有函数批量生成 Frida Hook 代码
  - 异步处理，不阻塞 IDA Pro 界面
  - 可选择目标 Frida 版本

### 2. MCP 工具接口模式

通过 ida-pro-mcp 工具可以调用以下 JSON-RPC 方法：

#### `generate_frida_hook_for_function`
- **参数**: 
  - `function_address` (字符串) - 函数地址
  - `frida_version` (字符串, 可选) - Frida版本 (默认: "17.x")
- **返回**: 生成的 Frida Hook 代码字符串
- **说明**: 为指定地址的函数生成 Frida Hook 代码

#### `generate_frida_hook_for_current_function`
- **参数**: 
  - `frida_version` (字符串, 可选) - Frida版本 (默认: "17.x")
- **返回**: 生成的 Frida Hook 代码字符串
- **说明**: 为当前选中的函数生成 Frida Hook 代码

#### `generate_frida_hooks_batch`
- **参数**: 
  - `function_addresses` (字符串数组) - 函数地址列表
  - `frida_version` (字符串, 可选) - Frida版本 (默认: "17.x")
- **返回**: 包含每个函数处理结果的数组
- **说明**: 批量为多个函数生成 Frida Hook 代码

#### `get_frida_version_info`
- **参数**: 无
- **返回**: Frida版本信息和差异对比
- **说明**: 获取支持的Frida版本详细信息

## 生成的 Frida 代码特性

### 自动解析功能
- **函数原型解析**: 自动解析函数参数类型和返回值类型
- **参数日志**: 为每个参数生成详细的日志输出
- **返回值日志**: 根据返回值类型生成相应的日志

### 生成的代码模板
```javascript
// Hook for function: function_name at 0x12345678
var function_name_addr = ptr("0x12345678");
Interceptor.attach(function_name_addr, {
    onEnter: function(args) {
        console.log("[+] Entering function_name");
        console.log("    arg0 (int param1): " + args[0]);
        console.log("    arg1 (char* param2): " + args[1]);
        // 在这里添加自定义的参数处理逻辑
    },
    onLeave: function(retval) {
        console.log("[+] Leaving function_name, return value (int): " + retval);
        // 在这里添加自定义的返回值处理逻辑
    }
});
```

## 使用示例

### 1. 通过 IDA Pro 插件使用
1. 打开 IDA Pro，加载目标文件
2. 选择要分析的函数
3. 通过菜单 `Edit/MCP/Generate Frida Hook (Current Function)` 或快捷键 `Ctrl-Alt-F`
4. 生成的代码保存到当前目录的 `frida_hooks.js` 文件中

### 2. 通过 MCP 工具使用

#### 生成单个函数的Hook（指定版本）
```python
# 使用 MCP 客户端调用 - Frida 17.x版本
result = client.call("generate_frida_hook_for_function", {
    "function_address": "0x401000",
    "frida_version": "17.x"
})
print(result)  # 打印生成的现代语法Frida Hook代码

# 使用 MCP 客户端调用 - Frida 16.x版本
result = client.call("generate_frida_hook_for_function", {
    "function_address": "0x401000", 
    "frida_version": "16.5"
})
print(result)  # 打印生成的传统语法Frida Hook代码
```

#### 批量生成多个函数的Hook
```python
# 批量生成，指定Frida版本
result = client.call("generate_frida_hooks_batch", {
    "function_addresses": ["0x401000", "0x401200", "0x401300"],
    "frida_version": "17.x"
})

for hook_result in result:
    if hook_result["success"]:
        print(f"Generated hook for {hook_result['name']}:")
        print(hook_result["hook_code"])
    else:
        print(f"Failed to generate hook for {hook_result['address']}: {hook_result['error']}")
```

#### 获取版本信息
```python
# 获取支持的Frida版本信息
version_info = client.call("get_frida_version_info")
print(f"Supported versions: {version_info['supported_versions']}")
print(f"Default version: {version_info['default_version']}")

# 查看版本差异
for version, details in version_info["version_differences"].items():
    print(f"\n{version}: {details['description']}")
    print(f"Features: {details['features']}")
```

## 技术实现细节

### 核心类：FridaCodeGenerator
- `__init__(output_file, frida_version)`: 初始化，支持版本指定
- `_parse_version(version_str)`: 解析版本字符串
- `generate_hook_for_function()`: 核心生成函数
- `_parse_function_args()`: 解析函数参数
- `_parse_return_type()`: 解析返回值类型
- `_generate_legacy_template()`: 生成传统版本代码模板
- `_generate_modern_template()`: 生成现代版本代码模板

### 版本检测逻辑
```python
def _parse_version(self, version_str):
    """解析版本字符串，返回版本元组"""
    try:
        if version_str.endswith('.x'):
            version_str = version_str[:-2]
        parts = version_str.split('.')
        major = int(parts[0]) if len(parts) > 0 else 16
        minor = int(parts[1]) if len(parts) > 1 else 0
        return (major, minor)
    except (ValueError, IndexError):
        return (16, 0)  # 默认为旧版本

# 版本判断
self.is_legacy_version = self._parse_version(frida_version) < (17, 0)
```

### 智能类型检测
代码生成器能智能识别参数类型并生成对应的处理代码：
- **字符串类型** (`char*`, `string`): 使用 `readUtf8String()` 读取
- **整数类型** (`int`, `long`, `dword`): 使用 `toInt32()` 转换
- **指针类型** (`pointer`, `*`): 显示地址并进行内存dump
- **其他类型**: 直接显示原始值

### 集成方式
- **插件模式**: 通过 `idaapi.action_handler_t` 和菜单系统集成
- **MCP 模式**: 通过 `@jsonrpc` 装饰器注册为 RPC 方法
- **版本支持**: 动态根据指定版本生成不同语法的代码

## 文件结构
```
mcp-plugin.py
├── FridaCodeGenerator 类
│   ├── 版本解析逻辑
│   ├── 传统模板生成 (_generate_legacy_template)
│   └── 现代模板生成 (_generate_modern_template)
├── MCP 工具函数
│   ├── generate_frida_hook_for_function (支持版本参数)
│   ├── generate_frida_hook_for_current_function (支持版本参数)
│   ├── generate_frida_hooks_batch (支持版本参数)
│   └── get_frida_version_info (新增)
├── 插件 Action Handlers
│   ├── GenerateFridaCurrentHandler
│   └── GenerateFridaAllHandler
└── 菜单注册
    └── register_frida_menu()
```

## 版本迁移指南

### 从 Frida 16.x 迁移到 17.x

1. **语法现代化**:
   - `var` → `const`/`let`
   - `function()` → 箭头函数
   - 字符串拼接 → 模板字面量

2. **API增强**:
   - 使用 `ArrayBuffer.wrap()` 进行高效内存访问
   - 利用增强的错误处理机制
   - 采用现代JavaScript特性

3. **代码生成差异**:
   ```javascript
   // 16.x 风格
   console.log("Value: " + args[0].toInt32());
   
   // 17.x 风格  
   console.log(`Value: ${args[0].toInt32()}`);
   ```

## 扩展建议

1. **增强参数解析**: 支持更复杂的函数原型解析和结构体识别
2. **模板定制**: 允许用户自定义 Frida Hook 模板
3. **条件Hook**: 支持生成带条件的 Frida Hook 代码
4. **内存分析**: 集成内存读写和结构体解析功能
5. **批量处理优化**: 支持按模块或函数类型过滤
6. **版本自动检测**: 自动检测目标设备的Frida版本
7. **代码优化**: 根据函数复杂度生成不同详细程度的Hook代码

## 注意事项

- 确保 IDA Pro 已正确加载目标文件
- 生成的 Frida 代码需要根据具体目标程序进行调整
- 批量生成时建议分批处理，避免内存占用过大
- 函数原型解析依赖于 IDA Pro 的分析结果，复杂函数可能需要手动调整
- **重要**: 使用前请确认目标环境的 Frida 版本，选择对应的代码生成模式
- Frida 17.x 的新特性需要相应版本的支持，在旧环境中可能不兼容

## 🔧 故障排除

### 常见问题

1. **版本不匹配**: 如果生成的代码在目标环境中报错，请检查Frida版本并重新生成
2. **语法错误**: 确保使用正确的版本参数（"16.x" 或 "17.x"）
3. **参数类型识别**: 复杂的函数原型可能需要手动调整生成的代码
4. **内存访问**: 指针参数的内存dump可能因权限问题失败，需要添加适当的检查
