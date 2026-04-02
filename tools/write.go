// write.go 实现了文件写入工具（WriteTool），允许 AI 代理创建、覆盖或编辑文件。
// 支持两种操作模式：
//   - 完整写入模式：提供 path 和 content，创建新文件或完全覆盖现有文件
//   - 部分编辑模式：提供 path、old_string 和 new_string，替换文件中的指定内容
package tools

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// WriteTool 是文件写入工具的实现，实现了 Tool 接口。
// 该工具支持创建新文件、覆盖已有文件，以及对文件进行部分内容替换。
type WriteTool struct{}

// WriteInput 定义了完整写入模式的输入参数结构。
type WriteInput struct {
	// Path 是目标文件路径（string 类型），支持绝对路径和相对路径。
	Path string `json:"path"`
	// Content 是要写入文件的完整内容（string 类型）。
	Content string `json:"content"`
}

// EditInput 定义了部分编辑模式的输入参数结构。
type EditInput struct {
	// Path 是目标文件路径（string 类型），支持绝对路径和相对路径。
	Path string `json:"path"`
	// OldString 是要在文件中查找并替换的原始文本（string 类型）。
	OldString string `json:"old_string"`
	// NewString 是用于替换 OldString 的新文本（string 类型）。
	NewString string `json:"new_string"`
}

// Name 返回工具的名称标识符 "write"。
// 返回值：string 类型的工具名称。
func (t *WriteTool) Name() string {
	return "write"
}

// Description 返回工具的功能描述。
// 返回值：string 类型的工具描述，告知 AI 模型该工具支持完整写入和部分编辑两种模式。
func (t *WriteTool) Description() string {
	return "Create or overwrite a file with the given content. Use old_string and new_string for partial edits."
}

// InputSchema 返回工具输入参数的 JSON Schema 定义。
// 返回值：map[string]interface{} 类型的 JSON Schema，定义了以下参数：
//   - path (string, 必填)：目标文件路径
//   - content (string, 可选)：完整写入的内容
//   - old_string (string, 可选)：要替换的原始文本
//   - new_string (string, 可选)：替换后的新文本
func (t *WriteTool) InputSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"path": map[string]interface{}{
				"type":        "string",
				"description": "The path to the file",
			},
			"content": map[string]interface{}{
				"type":        "string",
				"description": "The content to write (for create/overwrite)",
			},
			"old_string": map[string]interface{}{
				"type":        "string",
				"description": "The text to replace (for partial edit)",
			},
			"new_string": map[string]interface{}{
				"type":        "string",
				"description": "The replacement text (for partial edit)",
			},
		},
		"required": []string{"path"},
	}
}

// Execute 执行文件写入或编辑操作。
// 参数：
//   - input (json.RawMessage)：JSON 格式的输入参数
//
// 返回值：
//   - string：操作成功时返回成功消息
//   - error：操作失败时返回错误信息
//
// 执行逻辑：
//  1. 解析 JSON 参数为 map（因为参数组合灵活，不使用固定结构体）
//  2. 验证 path 参数是否存在
//  3. 如果路径是相对路径，转换为基于 cwd 的绝对路径
//  4. 判断操作模式：
//     - 如果提供了 old_string：进入部分编辑模式（读取文件 → 查找替换 → 写回文件）
//     - 否则：进入完整写入模式（创建必要的目录 → 写入内容到文件）
func (t *WriteTool) Execute(input json.RawMessage) (string, error) {
	// 将 JSON 输入解析为通用的 map 结构（因为两种模式参数不同）
	// args (map[string]interface{})：包含所有传入的键值对参数
	var args map[string]interface{}
	if err := json.Unmarshal(input, &args); err != nil {
		return "", fmt.Errorf("invalid input: %w", err)
	}

	// 从 map 中提取并验证 path 参数
	// path (string)：目标文件路径
	path, _ := args["path"].(string)
	if path == "" {
		return "", fmt.Errorf("path is required")
	}
	// 处理相对路径：如果不是绝对路径，则与 cwd 拼接
	if !filepath.IsAbs(path) {
		path = filepath.Join(cwd, path)
	}

	// 部分编辑模式：当提供了 old_string 参数时进入此分支
	// 该模式会读取现有文件内容，找到 old_string 并替换为 new_string
	if oldStr, ok := args["old_string"].(string); ok {
		// newStr (string)：替换后的新文本，可以为空（即删除 old_string）
		newStr, _ := args["new_string"].(string)
		// 读取原文件内容
		// data ([]byte)：文件原始字节数据
		data, err := os.ReadFile(path)
		if err != nil {
			return "", fmt.Errorf("read %s for edit: %w", path, err)
		}
		content := string(data)
		// 检查 old_string 是否存在于文件中
		if !strings.Contains(content, oldStr) {
			return "", fmt.Errorf("old_string not found in %s", path)
		}
		// 执行替换操作（仅替换第一次出现的位置）
		// strings.Replace 的第四个参数 1 表示只替换第一个匹配项
		content = strings.Replace(content, oldStr, newStr, 1)
		// 将修改后的内容写回文件，权限设置为 0644（所有者读写，其他人只读）
		if err := os.WriteFile(path, []byte(content), 0644); err != nil {
			return "", fmt.Errorf("write %s: %w", path, err)
		}
		return fmt.Sprintf("Successfully edited %s", path), nil
	}

	// 完整写入模式：创建新文件或覆盖已有文件
	// content (string)：要写入的完整内容
	content, _ := args["content"].(string)
	// dir (string)：文件所在的目录路径
	dir := filepath.Dir(path)
	// 递归创建所有必要的父目录（如果不存在）
	// os.MkdirAll 会忽略已存在的目录，权限设置为 0755
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("create directory %s: %w", dir, err)
	}
	// 将内容写入文件，如果文件已存在会被完全覆盖
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		return "", fmt.Errorf("write %s: %w", path, err)
	}
	return fmt.Sprintf("Successfully wrote %s", path), nil
}
