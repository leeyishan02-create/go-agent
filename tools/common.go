// Package tools 包含了 go-agent 可以使用的各种工具实现，
// 包括文件读取（ReadTool）、文件写入（WriteTool）和网络搜索（SearchTool）。
// 这些工具实现了 main 包中定义的 Tool 接口，可以被 AI 模型通过 function calling 自动调用。
package tools

// cwd 存储当前工作目录的路径。
// 所有文件操作工具（如 ReadTool、WriteTool）在处理相对路径时，
// 会以此目录为基准将相对路径转换为绝对路径。
var cwd string

// SetCwd 设置当前工作目录。
// 参数 dir (string) 是要设置的工作目录路径。
// 该函数在程序初始化时被调用（通常在 query.go 的 GetTools 或 GetToolsWithSearch 中），
// 确保工具在执行文件操作时使用正确的基准目录。
func SetCwd(dir string) {
	cwd = dir
}
