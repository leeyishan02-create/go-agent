// Package main 是 go-agent 应用程序的入口包。
// go-agent 是一个基于 OpenAI 兼容 API 的 AI 对话代理，
// 支持命令行交互模式和 Web UI 模式，具备文件读写和网络搜索等工具调用能力。
//
// main.go 文件包含程序的入口函数 main()，负责解析命令行参数并启动对应的运行模式：
//   - 默认模式：命令行交互式对话（REPL）
//   - --web 模式：启动 Web UI 服务器
//   - -p/--print 模式：执行单次查询后退出
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"
)

// main 是程序的入口函数。
// 该函数执行以下操作：
//  1. 解析命令行参数（--web 和 --port），flag.Bool 返回 *bool 类型，flag.Int 返回 *int 类型
//  2. 创建一个可通过 Ctrl+C (SIGINT) 或 SIGTERM 信号取消的上下文（context.Context）
//  3. 根据参数决定运行模式：
//     - Web 模式：启动 HTTP 服务器提供 Web UI 界面
//     - 单次查询模式（-p/--print）：执行一次查询后退出
//     - 交互模式：启动 REPL 循环，等待用户输入
func main() {
	// 定义命令行标志参数
	// webMode (*bool)：是否启动 Web UI 模式
	webMode := flag.Bool("web", false, "Start web UI server")
	// port (*int)：Web 服务器监听端口，默认为 8080
	port := flag.Int("port", 8080, "Web server port")
	// 解析命令行参数
	flag.Parse()

	// 创建可取消的上下文，监听 SIGINT（Ctrl+C）和 SIGTERM 信号
	// signal.NotifyContext 返回一个 context.Context 和一个取消函数
	// 当收到指定信号时，ctx 会自动取消
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel() // 确保程序退出时释放信号监听资源

	// Web UI 模式：启动 HTTP 服务器
	if *webMode {
		fmt.Printf("Starting web server on port %d (Ctrl+C to stop)\n", *port)
		// 创建一个带缓冲的错误 channel，用于接收服务器启动错误
		errCh := make(chan error, 1)
		// 在 goroutine 中启动 Web 服务器，避免阻塞主 goroutine
		go func() {
			errCh <- StartWebServer(*port)
		}()

		// 阻塞等待上下文取消（即收到终止信号）
		<-ctx.Done()
		fmt.Println("\nShutting down...")
		return
	}

	// 创建 API 客户端和工具列表
	// client (*Client)：封装了与 LLM API 的通信逻辑
	client := NewClient()
	// toolList ([]Tool)：可用的工具列表（如读文件、写文件等）
	toolList := GetTools()

	// 打印启动信息，显示当前使用的模型和 API 地址
	fmt.Printf("go-agent (model: %s, api: %s)\n", client.Model, client.BaseURL)
	fmt.Println("Type your message, or /quit to exit. Use --web for web UI.")
	fmt.Println()

	// 单次查询模式（-p 或 --print）：从命令行参数获取提示词，执行一次查询后退出
	// 注意：这里直接检查 os.Args 而不是通过 flag 包，因为该参数需要消费后续所有参数
	if len(os.Args) > 1 && (os.Args[1] == "-p" || os.Args[1] == "--print") {
		// args ([]string)：-p 参数后面的所有参数，组合为提示词
		args := os.Args[2:]
		if len(args) == 0 {
			fmt.Fprintln(os.Stderr, "Usage: go-agent -p <prompt>")
			os.Exit(1)
		}
		// 将所有参数用空格连接成一个完整的提示词
		prompt := strings.Join(args, " ")
		// 调用 Query 执行查询，结果直接打印到标准输出
		if err := Query(ctx, client, toolList, prompt); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		return
	}

	// 交互式 REPL 模式：循环读取用户输入并处理查询
	// scanner (*bufio.Scanner)：用于逐行读取标准输入
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ") // 打印输入提示符
		// 读取一行输入，如果到达 EOF（如管道输入结束）则退出
		if !scanner.Scan() {
			break
		}
		// line (string)：去除首尾空白后的用户输入
		line := strings.TrimSpace(scanner.Text())
		// 跳过空行
		if line == "" {
			continue
		}
		// 检查退出命令
		if line == "/quit" || line == "/exit" {
			break
		}

		// 执行用户查询，支持工具调用和流式输出
		if err := Query(ctx, client, toolList, line); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		}
		fmt.Println() // 在每次回复后打印空行以分隔输出
	}
}
