document.addEventListener("DOMContentLoaded", () => {
    const chatBox = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const resetBtn = document.getElementById("reset-btn");
    const loading = document.getElementById("loading");

    // 初始化 mermaid，并确保 mindmap 插件已注册
    // 某些版本的 mermaid-mindmap 插件在加载时已自动注册，
    // 如果我们再次手动调用 registerExternalDiagrams 会抛出异常，
    // 进而导致后续 mermaid.render 无法执行。
    // 因此，这里增加存在性检查并使用 try-catch 进行防御，
    // 确保即便注册失败也不会影响主逻辑。
    try {
        if (window.mermaidMindmap && typeof mermaid.registerExternalDiagrams === 'function') {
            mermaid.registerExternalDiagrams([window.mermaidMindmap]);
        }
    } catch (e) {
        console.warn('Mermaid mindmap plugin registration skipped:', e);
    }

    mermaid.initialize({ startOnLoad: false, theme: 'neutral', securityLevel: 'loose' });

    // 保存初始欢迎信息，用于重置对话
    const initialChatHTML = chatBox.innerHTML;

    const resetChat = () => {
        chatBox.innerHTML = initialChatHTML;
        userInput.value = "";
    };

    const showLoading = (show) => {
        loading.style.display = show ? 'block' : 'none';
        sendBtn.disabled = show;
    };

    const sendMessage = async () => {
        const query = userInput.value.trim();
        if (!query) return;

        appendMessage(query, "user");
        userInput.value = "";
        showLoading(true);

        try {
            const response = await fetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ message: query }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            appendMessage(data.response, "bot");

        } catch (error) {
            console.error("Error:", error);
            appendMessage("抱歉，处理您的请求时出错，请查看控制台了解详情。", "bot");
        } finally {
            showLoading(false);
        }
    };

    const appendMessage = (content, type) => {
        const messageWrapper = document.createElement("div");
        messageWrapper.className = "message";

        const messageBubble = document.createElement("div");
        messageBubble.className = `message-bubble ${type === 'user' ? 'user-message' : 'bot-message'}`;

        // ---------- Mermaid 渲染处理 ----------
        // 简化版：只查找 ```mermaid ... ``` 代码块
        const mermaidBlockRegex = /```mermaid([\s\S]*?)```/i;
        const match = content.match(mermaidBlockRegex);

        if (type === 'bot' && match) {
            const mermaidSource = match[1].trim();
            const summaryText = content.replace(match[0], '').trim();

            // 创建容器：图 + 总结
            const graphDiv = document.createElement('div');
            graphDiv.className = 'mermaid';
            // 将源码直接放入 mermaid 容器，后续由 mermaid.init 解析
            graphDiv.textContent = mermaidSource;
            messageBubble.appendChild(graphDiv);

            if (summaryText) {
                const summaryDiv = document.createElement('div');
                summaryDiv.className = 'mermaid-summary';
                summaryDiv.innerHTML = marked.parse(summaryText);
                messageBubble.appendChild(summaryDiv);
            }

            // 异步渲染，确保元素已插入 DOM
            setTimeout(() => {
                try {
                    // 一些 CDN 版本的 mindmap 插件会自动注册，也可能挂载为 window.mindmap
                    if (typeof mermaid.registerExternalDiagrams === 'function') {
                        const mindmapPlugin = window.mermaidMindmap || window.mindmap;
                        if (mindmapPlugin) {
                            // registerExternalDiagrams 会抛出错误若重复注册，故 catch 内自行忽略
                            try {
                                mermaid.registerExternalDiagrams([mindmapPlugin]);
                            } catch (_) {}
                        }
                    }
                    mermaid.init(undefined, graphDiv);
                } catch (e) {
                    console.error('Mermaid render error:', e);
                    // 渲染失败时，将源码当作文本展示
                    graphDiv.innerHTML = `<pre><code>${mermaidSource}</code></pre>`;
                }
            }, 100);

        } else if (type === 'bot') {
            // 对于不含 Mermaid 的机器人消息，正常使用 marked.js 渲染
            messageBubble.innerHTML = marked.parse(content);
        } else {
            // 用户消息直接作为文本展示
            messageBubble.textContent = content;
        }

        messageWrapper.appendChild(messageBubble);
        chatBox.appendChild(messageWrapper);
        chatBox.scrollTop = chatBox.scrollHeight;
    };

    sendBtn.addEventListener("click", sendMessage);
    resetBtn.addEventListener("click", resetChat);
    userInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Remove the initial welcome message from JS, as it's now in the HTML
}); 