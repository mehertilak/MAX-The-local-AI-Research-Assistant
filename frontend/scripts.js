class ChatApp {
    constructor() {
        // State Management
        this.ragMode = false;
        this.voiceMode = false;
        this.darkMode = false;
        this.currentMode = 'chat';
        this.history = [];
        this.recordedChunks = [];
        this.timerInterval = null;
        this.tutorialSeen = localStorage.getItem('tutorialSeen') === 'true'; // Track if tutorial seen
        this.isRecording = false;
        this.uploadedImage = null; // Variable to store the uploaded image file

        // DOM Elements
        this.chatInput = document.getElementById('chatInput');
        this.chatMessages = document.getElementById('chatMessages');
        this.welcomeMessage = document.getElementById('welcomeMessage');
        this.tutorialOverlay = document.getElementById('tutorialOverlay'); // Tutorial overlay
        this.tutorialToggleButton = document.getElementById('tutorialToggleButton'); // Tutorial toggle button
        this.liveStatusCircle = document.getElementById('liveStatusCircle');
        this.micBtn = document.getElementById('micBtn'); // Mic button
        this.voiceCommOverlay = document.getElementById('voiceCommOverlay'); // Voice overlay

        // Initialize
        this.initializeEventListeners();
        this.initializeTheme();

        // Show tutorial if not seen before (initially disabled for toggle button implementation)
        // if (!this.tutorialSeen) {
        //     this.showTutorial();
        // }
    }

    // ========================
    // Core Initialization
    // ========================
    initializeEventListeners() {
        // Message Handling
        this.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.handleSendMessage();
        });

        // File Handling
        document.getElementById('paperclipBtn').addEventListener('click', () => {
            this.currentMode = 'rag';
            document.getElementById('fileInput').click();
        });
        document.getElementById('fileInput').addEventListener('change', (e) => this.handleFileUpload(e));

        // Image Handling
        document.getElementById('cameraBtn').addEventListener('click', () => {
            this.currentMode = 'image_chat';
            document.getElementById('imageInput').click();
        });
        document.getElementById('imageInput').addEventListener('change', (e) => this.handleImageUpload(e));

        // Voice Chat
        this.micBtn.addEventListener('click', () => this.toggleVoiceChat());
        document.getElementById('voiceCommClose').addEventListener('click', () => this.stopVoiceChat());

        // Theme
        document.getElementById('themeSwitch').addEventListener('click', () => this.toggleTheme());

        // Tutorial Toggle Button
        this.tutorialToggleButton.addEventListener('click', () => this.toggleTutorial());


        // QR Scanner
        document.getElementById('scannerBtn').addEventListener('click', () => this.startQrScanner());
        document.getElementById('qrScannerClose').addEventListener('click', () => this.stopQrScanner());

        // Video Handling
        document.getElementById('videoBtn').addEventListener('click', () => {
             if(this.currentMode === 'rag'){
                this.cleanupRag();
             }
            if(this.currentMode === 'web_crawler'){
                this.cleanupWeb();
            }
             if(this.currentMode === 'image_chat'){
                this.cleanupImage();
            }
            this.currentMode = 'live_chat';
            document.getElementById('videoOptionsOverlay').style.display = 'flex';
        });

        document.getElementById('startLiveVideoBtn').addEventListener('click', async () => {
            document.getElementById('liveRecorderOverlay').style.display = 'flex';
            // Call the new startLiveVideo function
            await this.startLiveVideo();
        });

        document.getElementById('videoOptionsClose').addEventListener('click', () => {
            document.getElementById('videoOptionsOverlay').style.display = 'none';
        });

        document.getElementById('liveRecorderClose').addEventListener('click', async () => {
            document.getElementById('liveRecorderOverlay').style.display = 'none';
             // Call the new stopLiveVideo function
            await this.stopLiveVideo();
             if(this.currentMode === 'rag'){
                this.cleanupRag();
            }
            if(this.currentMode === 'web_crawler'){
                this.cleanupWeb();
            }
            if(this.currentMode === 'image_chat'){
                this.cleanupImage();
            }
             this.currentMode = 'chat';
        });

        // Status Circle
        this.liveStatusCircle.addEventListener('click', () => this.toggleLiveStatus());

        // Tutorial Close Button
        document.getElementById('tutorialClose').addEventListener('click', () => this.hideTutorial());

        // Removed event listener for videoInput change
        // document.getElementById('videoInput').addEventListener('change', (e) => this.handleVideoUpload(e));
    }

    // ========================
    // Image Handling
    // ========================
   async handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Step 1: Show preview immediately
        const imageUrl = URL.createObjectURL(file);
        this.addMessage('user', `<img src="${imageUrl}" class="image-preview">`);


         this.showTypingIndicator();
        const formData = new FormData();
         formData.append('image', file);

         try {
             const response = await fetch('/upload_image', {
                method: 'POST',
                 body: formData
            });

             if (response.ok) {
                 const data = await response.json();
                if(data.success){
                     this.addMessage('ai', data.response);
                    this.addMessage('ai', `Image Caption: ${data.caption}`);
                     this.currentMode = 'image_chat';
                    this.uploadedImage = file; // Store the uploaded image file
                 }
                else{
                     this.addMessage('ai', `Image processing failed: ${data.error}`);
                 }

             }
             else{
                const errorData = await response.json()
                 this.addMessage('ai', `Image processing failed: ${errorData.error}`);
             }

         } catch (error) {
             this.addMessage('ai', 'Image processing failed');
         } finally {
             this.hideTypingIndicator();
             event.target.value = '';
        }
    }


    // ========================
    // API Communication
    // ========================
    async postData(url, data) {
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
          if (!response.ok) {
              const errorData = await response.json()
              throw new Error(errorData.error);
          }
           return await response.json();
        }
        catch (error) {
            console.error('API Error:', error);
             this.addMessage('ai', `Error: ${error.message}`);
            return null;
        }
    }

    // ========================
    // Message Handling
    // ========================
    async handleSendMessage() {
        const message = this.chatInput.value.trim();
        if (!message) return;


        this.addMessage('user', message);
        this.chatInput.value = '';
        this.welcomeMessage.classList.add('fade-out');
        this.showTypingIndicator();
        try {
            let response;
            if(this.currentMode === 'chat'){
                response = await this.postData('/chat', {
                    message: message
                });
                if (response?.response) {
                    this.addMessage('ai', response.response);
                }
            }
            else if (this.currentMode === 'rag'){
                if(message.toLowerCase() === 'exit'){
                    this.cleanupRag();
                    this.addMessage('ai', `Exiting RAG mode.`);
                    this.currentMode = 'chat'
                    return;
                }
                response = await this.postData('/rag_query', {
                    question: message
                });
                if (response?.response) {
                    this.addMessage('ai', response.response, response.document_name);
                }
            }
            else if (this.currentMode === 'web_crawler'){
                 if(message.toLowerCase() === 'exit'){
                    this.cleanupWeb();
                    this.addMessage('ai', `Exiting Web Crawler mode.`);
                    this.currentMode = 'chat'
                    return;
                }
                response = await this.postData('/web_query', {
                    question: message
                });
                 if (response?.response) {
                    this.addMessage('ai', response.response, response.document_name);
                 }
            }
            else if (this.currentMode === 'image_chat'){
                 if(message.toLowerCase() === 'exit'){
                     this.cleanupImage();
                     this.addMessage('ai', `Exiting Image Chat mode.`);
                      this.currentMode = 'chat'
                     return;
                 }

                 let response;
                 const formData = new FormData(); // Always create FormData in image_chat mode
                 formData.append('question', message);


                 try{
                     const response = await fetch('/image_query', {
                         method: 'POST',
                         body: formData, // Send FormData with just question now
                         // Content-Type is AUTOMATICALLY set by fetch for FormData
                     });
                     if (response.ok) {
                         const data = await response.json();
                         if(data.success){
                             if (data.image_path) {
                                  // Step 2: Display detected/pointed images
                                this.addMessage('ai', `<img src="${data.image_path}" class="image-preview">`);
                             }
                             else{
                                 this.addMessage('ai', data.response)
                             }
                         }
                         else{
                             this.addMessage('ai', `Image processing failed: ${data.error}`);
                         }
                     }
                     else{
                         const errorData = await response.json()
                         this.addMessage('ai', `Image processing failed: ${errorData.error}`);
                     }
                 }
                 catch (error) {
                     this.addMessage('ai', `Error: ${error.message}`);
                 }
            }
        }
        finally {
            this.hideTypingIndicator();
        }
    }

    // ========================
    // File Handling
    // ========================
    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        this.showTypingIndicator();

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                if(data.success){
                   this.ragMode = true;
                  this.addMessage('ai', `I understand the details in the document, lets talk about it`);
                }
                else{
                   const errorData = await response.json()
                   this.addMessage('ai', `File Upload Failed: ${errorData.error}`);
                }
            }
             else{
                   const errorData = await response.json()
                   this.addMessage('ai', `File Upload Failed: ${errorData.error}`);
              }
        }
        catch (error) {
            this.addMessage('ai', 'File upload failed');
        }
        finally {
            this.hideTypingIndicator();
            event.target.value = '';
              if(this.currentMode === 'image_chat'){
                this.cleanupImage();
            }
        }
    }


    // ========================
    // Video Handling
    // ========================
   async startLiveVideo() {
        try {
            const response = await this.postData('/start_live_chat', {});
            if (!response || !response.success) {
                this.addMessage('ai', 'Failed to start live chat.');
                return;
            }

            // Add Loading indication as the camera stream is not available.
            document.getElementById('liveVideoPreview').style.display = 'none'; // Hide preview
            document.getElementById('loadingIndicator').style.display = 'block'; // Show loading

            this.liveStatusCircle.classList.add('active');
            this.isRecording = true;

        } catch (error) {
            console.error('Video error:', error);
            this.addMessage('ai', 'Failed to start live chat.');
        }
    }

    async stopLiveVideo() {
        try {
            const response = await this.postData('/stop_live_chat', {});
            if (!response || !response.success) {
                this.addMessage('ai', 'Failed to stop live chat.');
                return;
            }
             document.getElementById('liveVideoPreview').style.display = 'block'; // Show preview
             document.getElementById('loadingIndicator').style.display = 'none'; // Hide loading
            this.liveStatusCircle.classList.remove('active');
            this.isRecording = false;
        } catch (error) {
            console.error('Video error:', error);
            this.addMessage('ai', 'Failed to stop live chat.');
        }
    }

    // ========================
    // Live Status Circle
    // ========================
    toggleLiveStatus(){
        if(this.isRecording){
            this.liveStatusCircle.classList.remove('active');
            this.stopLiveVideo()
        }
        else{
           this.liveStatusCircle.classList.add('active');
           this.startLiveVideo()
        }
    }
    // ========================
    // Voice Chat System
    // ========================
    async toggleVoiceChat() {
        if (this.voiceMode) {
            // Stop voice chat
            this.voiceMode = false;
            this.voiceCommOverlay.style.display = 'none';
            this.micBtn.classList.remove('active'); // Deactivate the mic button

            try {
                const response = await this.postData('/stop_voice_chat', {});
                if (!response?.success) {
                    this.addMessage('ai', 'Failed to stop voice chat.');
                }
            } catch (error) {
                console.error("Error stopping voice chat:", error);
                this.addMessage('ai', 'Failed to stop voice chat.');
            }
        } else {
            // Start voice chat
            this.voiceMode = true;
            this.voiceCommOverlay.style.display = 'flex';
            this.micBtn.classList.add('active'); // Activate the mic button

            try {
                const response = await this.postData('/start_voice_chat', {});
                if (!response?.success) {
                    this.addMessage('ai', 'Failed to start voice chat.');
                    this.voiceMode = false; // Reset state
                    this.voiceCommOverlay.style.display = 'none'; // Hide the overlay if start fails
                    this.micBtn.classList.remove('active'); //Deactivate the mic button
                }
            } catch (error) {
                console.error("Error starting voice chat:", error);
                this.addMessage('ai', 'Failed to start voice chat.');
                this.voiceMode = false; // Reset state
                this.voiceCommOverlay.style.display = 'none'; // Hide the overlay if start fails
                this.micBtn.classList.remove('active'); //Deactivate the mic button
            }
        }
    }

    stopVoiceChat() {
         this.voiceMode = false;
        this.voiceCommOverlay.style.display = 'none';
        this.micBtn.classList.remove('active'); // Deactivate the mic button

         this.postData('/stop_voice_chat', {})
    }

    // ========================
    // RAG Mode Cleanup
    // ========================
     async cleanupRag(){
            try {
                await this.postData('/cleanup_rag',{});
                this.ragMode = false
            } catch (error) {
                console.error('RAG cleanup failed',error)
            }
     }

    // ========================
    // Web Crawler Mode Cleanup
    // ========================
     async cleanupWeb(){
         try {
            await this.postData('/cleanup_web',{});
            this.currentMode = 'chat';
        } catch (error) {
            console.error('Web crawler cleanup failed',error)
        }
    }
    // ========================
    // Image Chat Mode Cleanup
    // ========================
    async cleanupImage(){
         try {
             await this.postData('/cleanup_image',{});
             this.currentMode = 'chat';
         } catch (error) {
             console.error('Image chat cleanup failed',error)
         }
     }
    // ========================
    // QR Scanner
    // ========================
        startQrScanner() {
        document.getElementById('qrScannerOverlay').style.display = 'flex';
        this.html5QrCode = new Html5Qrcode('qr-scanner-view');
        
          let scanningActive = true; // New flag to prevent multiple triggers

        this.html5QrCode.start({
                facingMode: 'environment'
            }, {
                fps: 10,
                qrbox: 250
            },
            async (decodedText) => { // Use arrow function here
                 if (!scanningActive) return;
                 scanningActive = false; // Prevent further triggers

                document.getElementById('qrResultDisplay').textContent = decodedText;


                  // Check if the decoded text is a URL
                 if (this.isValidUrl(decodedText)) {
                       this.stopQrScanner();  // Close the popup immediately after URL detection
                    if(this.currentMode === 'web_crawler'){
                        await this.cleanupWeb();
                    }
                    if(this.currentMode === 'image_chat'){
                        await this.cleanupImage();
                    }
                     this.currentMode = 'web_crawler';

                      this.showTypingIndicator()
                    try {
                          const response = await this.postData('/crawl_website', {
                             url: decodedText
                            });
                         if(response){
                            this.addMessage('ai', `I saw through the veils of the Internet, Now I know it's secrets. Let's talk about it`);
                         }
                    } catch (error) {
                        this.addMessage('ai', `Web crawling failed ${error}`)
                    }
                    finally{
                        this.hideTypingIndicator()
                        
                    }


                 } else {
                     document.getElementById('qrResultDisplay').textContent = "Not a valid URL";
                      setTimeout(() => {
                            this.stopQrScanner()
                        }, 1000)
                 }

            },
             errorMessage => console.warn(errorMessage)
         );
     }

    stopQrScanner() {
        document.getElementById('qrScannerOverlay').style.display = 'none';
        if (this.html5QrCode) {
            this.html5QrCode.stop().then(()=>{
                 this.html5QrCode = null; // Clean up instance
            })
        }

    }
     isValidUrl(string) {
        try {
          new URL(string);
          return true;
        } catch (_) {
          return false;
        }
      }

    // ========================
    // UI Components
    // ========================
    addMessage(sender, message, sourceDoc = null) {
        const bubble = document.createElement('div');
        bubble.className = `message-bubble ${sender}`;
        bubble.innerHTML = sourceDoc ?
            `${message} <br> <span class="source-doc">Source: ${sourceDoc}</span>` :
            message;

        this.chatMessages.appendChild(bubble);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    showTypingIndicator() {
        const typing = document.createElement('div');
        typing.className = 'typing-indicator';
        for (let i = 0; i < 3; i++) {
            typing.appendChild(document.createElement('div')).className = 'typing-dot';
        }
        this.chatMessages.appendChild(typing);
    }

    hideTypingIndicator() {
        document.querySelector('.typing-indicator')?.remove();
    }

    // ========================
    // Theme Management
    // ========================
    initializeTheme() {
        this.darkMode = localStorage.getItem('theme') === 'dark';
        document.body.classList.toggle('dark-mode', this.darkMode);
        document.querySelector('.theme-switch-icon').textContent = this.darkMode ? '🌙' : '🔅';
        localStorage.setItem('theme', this.darkMode ? 'dark' : 'light');
    }

    toggleTheme() {
        this.darkMode = !this.darkMode;
        document.body.classList.toggle('dark-mode');
        document.querySelector('.theme-switch-icon').textContent = this.darkMode ? '🌙' : '🔅';
        localStorage.setItem('theme', this.darkMode ? 'dark' : 'light');
    }


    // ========================
    // Tutorial Management
    // ========================
    showTutorial() {
        this.tutorialOverlay.classList.add('active');
    }

    hideTutorial() {
        this.tutorialOverlay.classList.remove('active');
        localStorage.setItem('tutorialSeen', 'true'); // Mark tutorial as seen
    }

    toggleTutorial() {
        if (this.tutorialOverlay.classList.contains('active')) {
            this.hideTutorial();
        } else {
            this.showTutorial();
        }
    }
}

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    window.chatApp = new ChatApp();
});