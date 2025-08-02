document.addEventListener('DOMContentLoaded', () => {
  const chatBox = document.getElementById('chat-box');
  const userInput = document.getElementById('user-input');
  const sendBtn = document.getElementById('send-btn');
  const endBtn = document.getElementById('end-btn');
  const loading = document.getElementById('loading');

  let sessionId = null; // 会话 ID，用于多轮对话

  const showLoading = (show) => {
    loading.style.display = show ? 'block' : 'none';
    sendBtn.disabled = show;
    if (endBtn) endBtn.disabled = show;
  };

  const resetChat = () => {
    chatBox.innerHTML = '';
    sessionId = null;
    userInput.value = '';
  };

  const endDialogue = async () => {
    if (!sessionId) {
      resetChat();
      return;
    }
    showLoading(true);
    try {
      await fetch('/end_dialogue', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      });
    } catch (err) {
      console.error(err);
    } finally {
      showLoading(false);
      resetChat();
      appendMessage('（对话已结束，您可以开始新的对话）', 'bot');
    }
  };

  const appendMessage = (content, type) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'message';

    const bubble = document.createElement('div');
    bubble.className = `message-bubble ${type === 'user' ? 'user-message' : 'bot-message'}`;
    bubble.textContent = content;

    wrapper.appendChild(bubble);
    chatBox.appendChild(wrapper);
    chatBox.scrollTop = chatBox.scrollHeight;
  };

  const sendMessage = async () => {
    const text = userInput.value.trim();
    if (!text) return;

    appendMessage(text, 'user');
    userInput.value = '';
    showLoading(true);

    const url = sessionId ? '/continue_dialogue' : '/start_dialogue';
    const payload = sessionId ? { session_id: sessionId, message: text } : { message: text };

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (response.ok) {
        if (data.session_id) {
          sessionId = data.session_id;
        }
        appendMessage(data.response || data.message || '（无回复）', 'bot');
      } else {
        appendMessage(data.error || '发生错误，请稍后重试。', 'bot');
      }
    } catch (err) {
      console.error(err);
      appendMessage('网络错误，请检查连接。', 'bot');
    } finally {
      showLoading(false);
    }
  };

  sendBtn.addEventListener('click', sendMessage);
  endBtn.addEventListener('click', endDialogue);
  userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
});