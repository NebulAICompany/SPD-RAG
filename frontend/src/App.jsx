import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I am your AIris RRM Agent. You can upload files (.txt, .xlsx) and I will use them to answer your questions.' }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  
  // File Upload State
  const [selectedFile, setSelectedFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [selectedFilesForChat, setSelectedFilesForChat] = useState([])

  // Fetch file list on load
  useEffect(() => {
    fetchFiles();
  }, []);

  const fetchFiles = async () => {
    try {
      const response = await fetch('http://localhost:8001/files');
      if (response.ok) {
        const data = await response.json();
        setUploadedFiles(data.files || []);
      }
    } catch (error) {
      console.error("Error fetching files:", error);
    }
  };

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0])
  }

  const uploadFile = async () => {
    if (!selectedFile) return;
    setUploading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch('http://localhost:8001/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Upload failed");
      }

      const result = await response.json();
      setMessages(prev => [...prev, { role: 'assistant', content: `✅ Uploaded ${result.file_name} successfully!` }]);
      setSelectedFile(null);
      // document.getElementById("fileInput").value = ""; 
      fetchFiles(); // Refresh list

    } catch (error) {
      console.error("Upload error:", error);
      setMessages(prev => [...prev, { role: 'assistant', content: `❌ Upload failed: ${error.message}` }]);
    } finally {
      setUploading(false);
    }
  };

  const toggleFileSelection = (fileName) => {
    setSelectedFilesForChat(prev => {
      if (prev.includes(fileName)) {
        return prev.filter(f => f !== fileName);
      } else {
        return [...prev, fileName];
      }
    });
  };

  const sendMessage = async () => {
    if (!input.trim()) return

    const userMessage = { role: 'user', content: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setLoading(true)

    try {
      const response = await fetch('http://localhost:8001/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: input,
          session_id: 'rrm-session-1',
          selected_files: selectedFilesForChat.length > 0 ? selectedFilesForChat : null
        }),
      })

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`)
      }

      const data = await response.json()
      const aiMessage = { role: 'assistant', content: data.response }
      setMessages(prev => [...prev, aiMessage])
    } catch (error) {
      console.error("Error sending message:", error)
      setMessages(prev => [...prev, { role: 'assistant', content: "Error: Could not reach the agent." }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      <header className="chat-header">
        <h1>AIris RRM / Agent Graph</h1>
      </header>
      
      <div className="main-content">
        {/* Sidebar for Files */}
        <div className="sidebar">
          <h3>Knowledge Base</h3>
          
          <div className="upload-section">
            <input type="file" onChange={handleFileChange} />
            <button onClick={uploadFile} disabled={!selectedFile || uploading}>
              {uploading ? "Uploading..." : "Upload"}
            </button>
          </div>

          <div className="file-list">
             <h4>Available Files</h4>
             {uploadedFiles.length === 0 && <p style={{fontSize: '0.8em', color: '#666'}}>No files yet.</p>}
             {uploadedFiles.map((file, idx) => (
               <div key={idx} className="file-item">
                 <label>
                   <input 
                     type="checkbox" 
                     checked={selectedFilesForChat.includes(file)}
                     onChange={() => toggleFileSelection(file)}
                   />
                   <span title={file}>{file}</span>
                 </label>
               </div>
             ))}
          </div>
        </div>

        {/* Chat Window */}
        <div className="chat-interface">
          <div className="chat-window">
            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.role}`}>
                <div className="message-content">
                  <strong>{msg.role === 'user' ? 'You' : 'Agent'}:</strong>
                  <p>{msg.content}</p>
                </div>
              </div>
            ))}
            {loading && <div className="message assistant"><p>Thinking...</p></div>}
          </div>

          <div className="input-area">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                 if (e.key === 'Enter' && !e.shiftKey) {
                   e.preventDefault();
                   sendMessage();
                 }
              }}
              placeholder="Type your message..."
            />
            <button onClick={sendMessage} disabled={loading}>
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
