document.addEventListener("DOMContentLoaded", () => {
    const chatBox = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const resetBtn = document.getElementById("reset-btn");
    const loading = document.getElementById("loading");
    
    // 多模态功能元素
    const imageBtn = document.getElementById("image-btn");
    const imageInput = document.getElementById("image-input");
    const imagePreview = document.getElementById("image-preview");
    const previewImg = document.getElementById("preview-img");
    const removeImageBtn = document.getElementById("remove-image");
    
    // 图片相关变量
    let selectedImageData = null;

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
        // 清除图片
        clearSelectedImage();
    };
    
    // 图片处理函数
    const clearSelectedImage = () => {
        selectedImageData = null;
        imagePreview.style.display = 'none';
        previewImg.src = '';
        imageInput.value = '';
    };
    
    const handleImageSelection = (file) => {
        if (!file) return;
        
        // 检查文件类型
        if (!file.type.startsWith('image/')) {
            alert('请选择图片文件！');
            return;
        }
        
        // 检查文件大小（16MB限制）
        if (file.size > 16 * 1024 * 1024) {
            alert('图片文件过大，请选择小于16MB的图片！');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            selectedImageData = e.target.result;
            previewImg.src = selectedImageData;
            imagePreview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    };

    const showLoading = (show) => {
        loading.style.display = show ? 'block' : 'none';
        sendBtn.disabled = show;
    };

    const sendMessage = async () => {
        const query = userInput.value.trim();
        if (!query) return;

        // 检查是否有图片或文本
        if (!query && !selectedImageData) {
            alert("请输入文本或选择图片！");
            return;
        }

        // 显示用户消息（包括图片）
        appendMessage(query, "user", selectedImageData);
        
        // 准备发送的数据
        const requestData = { message: query };
        if (selectedImageData) {
            requestData.image = selectedImageData;
        }
        
        userInput.value = "";
        const currentImageData = selectedImageData; // 保存当前图片数据
        clearSelectedImage(); // 清除图片预览
        showLoading(true);

        try {
            const response = await fetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(requestData),
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

    const appendMessage = (content, type, imageData = null) => {
        const messageWrapper = document.createElement("div");
        messageWrapper.className = "message";

        const messageBubble = document.createElement("div");
        messageBubble.className = `message-bubble ${type === 'user' ? 'user-message' : 'bot-message'}`;
        
        // 如果是用户消息且有图片，先显示图片
        if (type === 'user' && imageData) {
            const imageElement = document.createElement('img');
            imageElement.src = imageData;
            imageElement.style.maxWidth = '200px';
            imageElement.style.maxHeight = '150px';
            imageElement.style.borderRadius = '10px';
            imageElement.style.marginBottom = '10px';
            imageElement.style.display = 'block';
            messageBubble.appendChild(imageElement);
        }

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
    
    // 图片上传相关事件监听器
    imageBtn.addEventListener("click", () => {
        imageInput.click();
    });
    
    imageInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) {
            handleImageSelection(file);
        }
    });
    
    removeImageBtn.addEventListener("click", clearSelectedImage);
    
    // 拖拽上传支持
    document.addEventListener("dragover", (e) => {
        e.preventDefault();
    });
    
    document.addEventListener("drop", (e) => {
        e.preventDefault();
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleImageSelection(files[0]);
        }
    });

    // Remove the initial welcome message from JS, as it's now in the HTML
}); 