package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"
)

func main() {
	webMode := flag.Bool("web", false, "Start web UI server")
	port := flag.Int("port", 8080, "Web server port")
	flag.Parse()

	if *webMode {
		if err := StartWebServer(*port); err != nil {
			fmt.Fprintf(os.Stderr, "Web server error: %v\n", err)
			os.Exit(1)
		}
		return
	}

	client := NewClient()
	toolList := GetTools()

	fmt.Printf("go-agent (model: %s, api: %s)\n", client.Model, client.BaseURL)
	fmt.Println("Type your message, or /quit to exit. Use --web for web UI.")
	fmt.Println()

	if len(os.Args) > 1 && (os.Args[1] == "-p" || os.Args[1] == "--print") {
		args := os.Args[2:]
		if len(args) == 0 {
			fmt.Fprintln(os.Stderr, "Usage: go-agent -p <prompt>")
			os.Exit(1)
		}
		prompt := strings.Join(args, " ")
		if err := Query(client, toolList, prompt); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		return
	}

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if line == "/quit" || line == "/exit" {
			break
		}

		if err := Query(client, toolList, line); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		}
		fmt.Println()
	}
}
