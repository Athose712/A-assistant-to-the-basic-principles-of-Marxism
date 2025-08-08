document.addEventListener("DOMContentLoaded", () => {
    const chatBox = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const resetBtn = document.getElementById("reset-btn");
    const endBtn = document.getElementById("end-btn");
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
        // 聊天模式没有真实的会话后端状态，这里仅统一按钮表现
        if (endBtn) endBtn.style.display = 'none';
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
            // 尝试将出题文本解析为结构化的题卡
            const maybeQuiz = tryParseQuiz(content);
            if (maybeQuiz && maybeQuiz.length > 0) {
                const quizContainer = renderQuizCards(maybeQuiz);
                messageBubble.appendChild(quizContainer);
            } else {
                // 对于不含 Mermaid 的机器人消息，正常使用 marked.js 渲染
                messageBubble.innerHTML = marked.parse(content);
            }
        } else {
            // 用户消息直接作为文本展示
            messageBubble.textContent = content;
        }

        messageWrapper.appendChild(messageBubble);
        chatBox.appendChild(messageWrapper);
        chatBox.scrollTop = chatBox.scrollHeight;
    };

    // ------------------ 出题解析：将纯文本解析为题卡 ------------------
    function tryParseQuiz(text) {
        // 粗略判定：包含“题目x：/选择题x：/题干：/正确答案：/解析：”等关键字
        const indicatorRegex = /(题目\s*\d+|选择题\s*\d+|判断题\s*\d+|简答题\s*\d+|题干\s*[:：]|正确答案\s*[:：]|参考答案\s*[:：]|解析\s*[:：])/;
        if (!indicatorRegex.test(text)) return null;

        const lines = text.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
        if (lines.length === 0) return null;

        // 找到每道题的起始行索引
        const titleRegex = /^(?:题目|选择题|判断题|简答题)\s*\d+\s*[:：]?/;
        const indices = [];
        for (let i = 0; i < lines.length; i++) {
            if (titleRegex.test(lines[i])) indices.push(i);
        }
        if (indices.length === 0) {
            // 如果没有显式题号，尝试把整体当作一题处理
            indices.push(0);
        }
        indices.push(lines.length); // 便于切片

        const questions = [];
        for (let k = 0; k < indices.length - 1; k++) {
            const start = indices[k];
            const end = indices[k + 1];
            const chunk = lines.slice(start, end);
            if (chunk.length === 0) continue;

            let title = chunk[0].replace(/\s+/g, '');
            let stem = '';
            const options = [];
            let answer = '';
            let explanation = '';

            const stemRegex = /^题干\s*[:：]\s*(.*)$/;
            const optionRegex = /^[A-DＡ-Ｄ]\s*[\.．、]\s*(.*)$/; // 兼容全角
            const answerRegex = /^(?:正确?答案|参考答案)\s*[:：]\s*(.*)$/;
            const explainRegex = /^(?:解析|答案解析|解答|讲解)\s*[:：]\s*(.*)$/;

            for (let i = 0; i < chunk.length; i++) {
                const line = chunk[i];
                // 跳过标题本身
                if (i === 0 && titleRegex.test(line)) continue;

                let m;
                if ((m = line.match(stemRegex))) {
                    stem = m[1];
                    continue;
                }
                if ((m = line.match(optionRegex))) {
                    const prefixMatch = line.match(/^[A-DＡ-Ｄ]/);
                    const prefix = prefixMatch ? prefixMatch[0].replace(/[Ａ-Ｄ]/, c => String.fromCharCode(c.charCodeAt(0) - 65248)) : '';
                    const text = line.replace(/^[A-DＡ-Ｄ]\s*[\.．、]\s*/, '');
                    options.push(`${prefix}. ${text}`);
                    continue;
                }
                if ((m = line.match(answerRegex))) {
                    answer = m[1];
                    continue;
                }
                if ((m = line.match(explainRegex))) {
                    // 解析可能多行，收集到下一条关键线或块末尾
                    const parts = [m[1]];
                    for (let j = i + 1; j < chunk.length; j++) {
                        const nxt = chunk[j];
                        if (stemRegex.test(nxt) || optionRegex.test(nxt) || answerRegex.test(nxt) || titleRegex.test(nxt)) break;
                        parts.push(nxt);
                        i = j; // 前移游标
                    }
                    explanation = parts.join('\n');
                    continue;
                }
            }

            // 若未抓到题干，退化为去标题后的首行
            if (!stem) {
                stem = chunk.slice(1).find(l => !optionRegex.test(l) && !answerRegex.test(l) && !explainRegex.test(l)) || '';
            }

            questions.push({ title, stem, options, answer, explanation });
        }

        return questions;
    }

    function renderQuizCards(questions) {
        const container = document.createElement('div');
        container.className = 'quiz-list';

        questions.forEach((q, idx) => {
            const card = document.createElement('div');
            card.className = 'quiz-card';

            const header = document.createElement('div');
            header.className = 'quiz-header';
            header.textContent = q.title || `题目${idx + 1}`;
            card.appendChild(header);

            if (q.stem) {
                const stem = document.createElement('div');
                stem.className = 'quiz-stem';
                stem.textContent = q.stem;
                card.appendChild(stem);
            }

            if (q.options && q.options.length > 0) {
                const optWrap = document.createElement('ul');
                optWrap.className = 'quiz-options';
                q.options.forEach(op => {
                    const li = document.createElement('li');
                    li.textContent = op;
                    optWrap.appendChild(li);
                });
                card.appendChild(optWrap);
            }

            const hasAnswer = q.answer && q.answer.trim().length > 0;
            const hasExplain = q.explanation && q.explanation.trim().length > 0;

            if (hasAnswer || hasExplain) {
                const actionBar = document.createElement('div');
                actionBar.className = 'quiz-actions';
                const toggleBtn = document.createElement('button');
                toggleBtn.className = 'quiz-toggle-btn';
                toggleBtn.textContent = '显示解析';
                actionBar.appendChild(toggleBtn);
                card.appendChild(actionBar);

                const detail = document.createElement('div');
                detail.className = 'quiz-detail hidden';

                if (hasAnswer) {
                    const ans = document.createElement('div');
                    ans.className = 'quiz-answer';
                    ans.innerHTML = `<span class="label">正确答案</span><span class="value">${escapeHTML(q.answer)}</span>`;
                    detail.appendChild(ans);
                }
                if (hasExplain) {
                    const exp = document.createElement('div');
                    exp.className = 'quiz-explain';
                    exp.innerHTML = `<span class="label">解析</span><div class="value">${escapeHTML(q.explanation).replace(/\n/g,'<br>')}</div>`;
                    detail.appendChild(exp);
                }
                card.appendChild(detail);

                toggleBtn.addEventListener('click', () => {
                    const isHidden = detail.classList.contains('hidden');
                    detail.classList.toggle('hidden');
                    toggleBtn.textContent = isHidden ? '隐藏解析' : '显示解析';
                });
            }

            container.appendChild(card);
        });

        return container;
    }

    function escapeHTML(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    sendBtn.addEventListener("click", sendMessage);
    resetBtn.addEventListener("click", resetChat);
    if (endBtn) {
        endBtn.addEventListener("click", resetChat);
    }
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