// read.go 实现了文件读取工具（ReadTool），允许 AI 代理读取指定文件的内容。
// 支持绝对路径和相对路径（相对于 cwd 当前工作目录）。
// 对于超大文件会自动截断内容，防止消耗过多内存或 token。
package tools

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// ReadTool 是文件读取工具的实现，实现了 Tool 接口。
// 该工具用于读取文本文件的内容并返回给 AI 模型。
type ReadTool struct{}

// ReadInput 定义了 ReadTool 的输入参数结构。
type ReadInput struct {
	// Path 是要读取的文件路径（string 类型），支持绝对路径和相对路径。
	// 如果是相对路径，会基于 cwd（当前工作目录）进行解析。
	Path string `json:"path"`
}

// Name 返回工具的名称标识符 "read"。
// 返回值：string 类型的工具名称。
func (t *ReadTool) Name() string {
	return "read"
}

// Description 返回工具的功能描述。
// 返回值：string 类型的工具描述，告知 AI 模型该工具支持读取文本文件。
func (t *ReadTool) Description() string {
	return "Read the contents of a file. Supports text files, and returns an error for binary files."
}

// InputSchema 返回工具输入参数的 JSON Schema 定义。
// 返回值：map[string]interface{} 类型的 JSON Schema，定义了以下参数：
//   - path (string, 必填)：要读取的文件路径
func (t *ReadTool) InputSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"path": map[string]interface{}{
				"type":        "string",
				"description": "The path to the file to read",
			},
		},
		"required": []string{"path"},
	}
}

// Execute 执行文件读取操作。
// 参数：
//   - input (json.RawMessage)：JSON 格式的输入参数，包含 "path" 字段
//
// 返回值：
//   - string：文件内容。如果文件超过 50000 字符，内容会被截断并附加提示信息
//   - error：读取失败时返回错误信息（如文件不存在、路径为目录等）
//
// 执行流程：
//  1. 解析 JSON 输入参数
//  2. 验证 path 参数是否为空
//  3. 如果路径是相对路径，转换为基于 cwd 的绝对路径
//  4. 检查文件状态（是否存在、是否为目录）
//  5. 读取文件内容
//  6. 如果内容超过 maxLen（50000 字符），截断并添加提示
func (t *ReadTool) Execute(input json.RawMessage) (string, error) {
	// 将 JSON 输入解析为 ReadInput 结构体
	var args ReadInput
	if err := json.Unmarshal(input, &args); err != nil {
		return "", fmt.Errorf("invalid input: %w", err)
	}

	// 验证必填参数
	if args.Path == "" {
		return "", fmt.Errorf("path is required")
	}

	// 处理相对路径：如果路径不是绝对路径，则与 cwd 拼接
	path := args.Path
	if !filepath.IsAbs(path) {
		path = filepath.Join(cwd, path)
	}

	// 获取文件信息，检查文件是否存在
	// info (os.FileInfo)：包含文件的元信息（大小、权限、是否为目录等）
	info, err := os.Stat(path)
	if err != nil {
		return "", fmt.Errorf("access %s: %w", path, err)
	}
	// 检查路径是否指向目录（不支持读取目录）
	if info.IsDir() {
		return "", fmt.Errorf("%s is a directory, not a file", path)
	}

	// 读取文件全部内容
	// data ([]byte)：文件的原始字节数据
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read %s: %w", path, err)
	}

	// 将字节数据转换为字符串
	content := string(data)
	// maxLen 定义最大返回字符数，防止文件过大时消耗过多 token
	const maxLen = 50000
	// 如果内容超过限制，截断并附加截断提示
	if len(content) > maxLen {
		content = content[:maxLen] + fmt.Sprintf("\n\n... (truncated, file is %d chars)", len(content))
	}

	return content, nil
}
