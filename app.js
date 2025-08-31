// app.js - Octopus demo controller (Hume EVI + Expressions)
// Интеграция с TalkingHead системой

(() => {
class OctoApp {
  constructor() {
    this.targetRate = 16000;
    this.audioCtx = null;
    this.userStream = null;
    this.audioOnlyStream = null;
    this.sourceNode = null;
    this.processor = null;
    this.analyser = null;
    this.eviWS = null;
    this.exprWS = null;
    this.exprCanStream = false;
    this.faceTimer = null;
    this._sinkGain = null;
    this.speaking = false;
    this._lipRAF = 0;
    this._speakTO = 0;
    this.lastUserAudioTs = 0;
    this.currentAudioSource = null;
    this.audioQueue = [];
    this.isPlayingAudio = false;
    this.lastFacePose = null;
    this.avatarReady = false;

    // Emotion smoothing parameters
    this.EMA_ALPHA = 0.25;
    this.emoValues = { 
      joy: 0, surprise: 0, fear: 0, anger: 0, 
      sadness: 0, disgust: 0, embarrassment: 0, pride: 0 
    };
    
    // Emotion keyword mapping
    this.EMOS = {
      joy: ['joy', 'amusement', 'excitement', 'satisfaction', 'happiness', 'positive', 'contentment', 'calmness', 'relief'],
      surprise: ['surprise', 'awe', 'astonishment', 'confusion', 'surprise (negative)'],
      fear: ['fear', 'anxiety', 'tension', 'worry', 'nervousness', 'apprehension'],
      anger: ['anger', 'annoyance', 'contempt', 'irritation', 'frustration', 'resentment'],
      sadness: ['sadness', 'disappointment', 'grief', 'boredom', 'loneliness', 'melancholy'],
      disgust: ['disgust', 'distaste', 'aversion'],
      embarrassment: ['embarrassment', 'shame', 'awkwardness'],
      pride: ['pride', 'confidence', 'concentration', 'determination', 'focus']
    };

    this.initUI();
    this.bindEvents();
    this.waitForAvatar();
    
    // Log initialization
    setTimeout(() => {
      this.log('UI готов - откройте консоль браузера (F12) для отладки');
    }, 100);
  }

  // ---------- UI Setup ----------
  initUI() {
    this.addDock();
    this.addHUD();
  }

  addDock() {
    const existing = document.getElementById('octoDock');
    if (existing) existing.remove();

    const dock = document.createElement('div');
    dock.id = 'octoDock';
    dock.style.cssText = [
      'position:fixed', 'right:12px', 'bottom:12px', 'z-index:100000',
      'width:360px', 'max-height:70vh', 'overflow:auto',
      'background:#141824', 'border:1px solid #21263a', 'border-radius:12px',
      'box-shadow:0 12px 40px rgba(0,0,0,.45)', 'padding:12px',
      'font:13px/1.4 system-ui,Segoe UI,Roboto,sans-serif', 'color:#e8edf8'
    ].join(';');
    
    dock.innerHTML = `
      <style>
        #octoDock .row{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0}
        #octoDock .btn{display:inline-flex;align-items:center;gap:8px;padding:8px 12px;border-radius:10px;background:#1d2336;border:1px solid #2a314a;color:#e8edf8;cursor:pointer;user-select:none}
        #octoDock .btn:disabled{opacity:.5;cursor:not-allowed}
        #octoDock .btn:hover:not(:disabled){background:#273254}
        #octoDock input, #octoDock textarea{width:100%;background:#0e1323;color:#e8edf8;border:1px solid #273254;border-radius:8px;padding:8px}
        #octoDock textarea{height:150px;resize:vertical}
        #octoDock h4{margin:0 0 8px 0;font-weight:700}
        #octoDock #camPreview{width:100%;height:160px;background:#000;border-radius:10px;display:flex;align-items:center;justify-content:center;margin-top:6px}
        #octoDock #camPreview video{width:100%;height:100%;object-fit:cover;border-radius:10px}
        #octoDock .status{padding:4px 8px;border-radius:6px;font-size:11px;margin-left:8px}
        #octoDock .status.ready{background:#1b4332;color:#40c057}
        #octoDock .status.error{background:#4a1e1e;color:#ff6b6b}
        #octoDock .status.loading{background:#3a3a1e;color:#ffd43b}
      </style>
      <h4>🐙 Octopus Control <span id="avatarStatus" class="status loading">Avatar Loading...</span></h4>
      <div class="row"><input id="apiKey" placeholder="Hume API Key (sk-...)" autocomplete="off"></div>
      <div class="row"><input id="configId" placeholder="Config ID (UUID or 'default')" autocomplete="off" value="default"></div>
      <div class="row">
        <button id="btnTestAPI" class="btn">🔑 Проверить API</button>
        <button id="btnTestMedia" class="btn">📷 Камера/Микрофон</button>
        <button id="btnTestSound" class="btn">🔊 Тест звука</button>
      </div>
      <div class="row">
        <button id="btnStart" class="btn" style="background:#214127;border-color:#2c6f39" disabled>▶️ НАЧАТЬ</button>
        <button id="btnStop" class="btn">⏹ СТОП</button>
      </div>
      <div id="camPreview"><span style="opacity:.6">Camera preview</span></div>
      <div class="row"><textarea id="log" spellcheck="false" readonly placeholder="System log..."></textarea></div>
    `;
    document.body.appendChild(dock);
  }

  addHUD() {
    const old = document.getElementById('emoHUD');
    if (old) old.remove();
    
    const wrap = document.createElement('div');
    wrap.id = 'emoHUD';
    wrap.style.cssText = [
      'position:fixed', 'left:12px', 'top:12px', 'width:240px',
      'background:rgba(12,12,20,.82)', 'backdrop-filter:blur(6px)',
      'padding:10px 12px', 'border-radius:12px',
      'font:12px/1.35 system-ui,sans-serif', 'color:#e8edf8',
      'box-shadow:0 6px 22px rgba(0,0,0,.35)', 'z-index:100002',
      'max-height:44vh', 'overflow:auto', 'pointer-events:none'
    ].join(';');
    
    const title = document.createElement('div');
    title.innerHTML = 'Emotions <small>(face+prosody)</small> <span id="hudTick" style="display:inline-block;width:8px;height:8px;background:#444;border-radius:50%;margin-left:6px;vertical-align:middle;"></span>';
    title.style.cssText = 'opacity:.9;margin:0 0 8px;font-weight:700;letter-spacing:.2px';
    wrap.appendChild(title);

    const createEmotionBar = (key) => {
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;align-items:center;gap:8px;margin:6px 0;';
      
      const label = document.createElement('div');
      label.textContent = key;
      label.style.cssText = 'width:106px;opacity:.95';
      
      const barWrap = document.createElement('div');
      barWrap.style.cssText = 'flex:1;height:8px;background:#2b2f45;border-radius:6px;overflow:hidden;';
      
      const bar = document.createElement('div');
      bar.id = 'bar-' + key;
      bar.style.cssText = 'height:100%;width:0%;background:#7bb2ff;transition:width .12s ease';
      
      const val = document.createElement('div');
      val.id = 'val-' + key;
      val.style.cssText = 'width:36px;text-align:right;opacity:.85';
      
      barWrap.appendChild(bar);
      row.appendChild(label);
      row.appendChild(barWrap);
      row.appendChild(val);
      return row;
    };

    Object.keys(this.emoValues).forEach(key => 
      wrap.appendChild(createEmotionBar(key))
    );
    
    document.body.appendChild(wrap);
  }

  bindEvents() {
    const safeHandler = (fn) => (e) => {
      e?.preventDefault?.();
      try {
        fn();
      } catch (err) {
        this.log('JS error: ' + err.message);
        console.error(err);
      }
    };

    // Get UI elements
    this.apiKeyEl = document.getElementById('apiKey');
    this.configEl = document.getElementById('configId');
    this.logEl = document.getElementById('log');
    this.btnStart = document.getElementById('btnStart');
    this.btnStop = document.getElementById('btnStop');
    this.avatarStatusEl = document.getElementById('avatarStatus');

    // Bind button events
    document.getElementById('btnTestSound')?.addEventListener('click', safeHandler(() => this.testBeep()));
    document.getElementById('btnTestMedia')?.addEventListener('click', safeHandler(() => this.testMedia()));
    document.getElementById('btnTestAPI')?.addEventListener('click', safeHandler(() => this.testAPI()));
    this.btnStart?.addEventListener('click', safeHandler(() => this.startAll()));
    this.btnStop?.addEventListener('click', safeHandler(() => this.stopAll()));

    // Unlock audio on first user interaction
    document.body.addEventListener('click', () => this.unlockAudio(), { once: true });
  }

  waitForAvatar() {
    // Listen for avatar ready event
    window.addEventListener('avatar:ready', () => {
      this.avatarReady = true;
      this.btnStart.disabled = false;
      this.avatarStatusEl.textContent = 'Avatar Ready';
      this.avatarStatusEl.className = 'status ready';
      this.log('✅ Avatar system готов');
    });

    // Listen for avatar error
    window.addEventListener('avatar:error', (e) => {
      this.avatarReady = false;
      this.avatarStatusEl.textContent = 'Avatar Error';
      this.avatarStatusEl.className = 'status error';
      this.log('❌ Avatar error: ' + (e.detail?.message || 'Unknown error'));
    });

    // Timeout fallback
    setTimeout(() => {
      if (!this.avatarReady) {
        this.log('⚠️ Avatar не готов, включаем fallback режим');
        this.btnStart.disabled = false;
        this.avatarStatusEl.textContent = 'Fallback Mode';
        this.avatarStatusEl.className = 'status error';
      }
    }, 10000);
  }

  log(msg) {
    console.log(`[OctoApp] ${msg}`);
    
    if (!this.logEl) {
      this.logEl = document.getElementById('log');
    }
    
    if (this.logEl) {
      const ts = new Date().toLocaleTimeString();
      this.logEl.value += `[${ts}] ${msg}\n`;
      this.logEl.scrollTop = this.logEl.scrollHeight;
    }
  }

  // ---------- Audio System ----------
  unlockAudio() {
    try {
      if (!this.audioCtx || this.audioCtx.state === 'closed') {
        this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (this.audioCtx.state === 'suspended') {
        this.audioCtx.resume();
      }
      this.log('🔓 Audio context unlocked');
    } catch (e) {
      console.error('Audio unlock error:', e);
    }
  }

  testBeep() {
    this.unlockAudio();
    if (!this.audioCtx) return;
    
    const osc = this.audioCtx.createOscillator();
    const gain = this.audioCtx.createGain();
    osc.frequency.value = 880;
    gain.gain.value = 0.06;
    osc.connect(gain).connect(this.audioCtx.destination);
    osc.start();
    setTimeout(() => osc.stop(), 140);
    this.log('🔊 Beep test completed');
  }

  async testMedia() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: true, 
        video: true 
      });
      
      const video = document.createElement('video');
      video.muted = true;
      video.playsInline = true;
      video.srcObject = stream;
      await video.play().catch(() => {});
      
      const camPreview = document.getElementById('camPreview');
      camPreview.innerHTML = '';
      camPreview.appendChild(video);
      
      this.log('📷 Media test OK - камера и микрофон готовы');
      
      setTimeout(() => {
        stream.getTracks().forEach(t => t.stop());
        camPreview.innerHTML = '<span style="opacity:.6">Camera preview</span>';
      }, 3000);
      
    } catch (e) {
      this.log('Media error: ' + e.message);
    }
  }

  async testAPI() {
    const key = (this.apiKeyEl?.value || '').trim();
    const cfg = (this.configEl?.value || '').trim();
    
    if (!key) {
      this.log('Введите Hume API key');
      return;
    }

    try {
      // Test Expressions API
      await this.testWebSocket(
        'wss://api.hume.ai/v0/stream/models?api_key=' + encodeURIComponent(key),
        'Expressions'
      );
      
      // Test EVI API
      await this.testWebSocket(
        'wss://api.hume.ai/v0/evi/chat?api_key=' + encodeURIComponent(key) + 
        '&config_id=' + encodeURIComponent(cfg || 'default'),
        'EVI'
      );
      
      this.log('✅ API tests passed');
      
    } catch (e) {
      this.log('API test failed: ' + e.message);
    }
  }

  testWebSocket(url, name) {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(url);
      const timeout = setTimeout(() => {
        ws.close();
        reject(new Error(`${name} timeout`));
      }, 5000);
      
      ws.onopen = () => {
        clearTimeout(timeout);
        ws.close();
        resolve();
      };
      
      ws.onerror = () => {
        clearTimeout(timeout);
        reject(new Error(`${name} connection error`));
      };
    });
  }

  // ---------- Main Control ----------
  async startAll() {
    const key = (this.apiKeyEl?.value || '').trim();
    const cfg = (this.configEl?.value || '').trim() || 'default';
    
    if (!key) {
      this.log('Введите Hume API key');
      return;
    }

    try {
      this.btnStart.disabled = true;
      this.log('🚀 Запускаем систему...');
      
      await this.ensureMedia();
      await this.openExpressions(key);
      await this.openEVI(key, cfg);
      
      this.streamMic();
      this.streamFace();
      
      this.log('▶️ Система запущена - говорите с осьминогом!');
      
    } catch (e) {
      this.log('Start error: ' + e.message);
      this.stopAll();
    }
  }

  stopAll() {
    this.log('⏹ Останавливаем систему...');

    // Stop audio queue
    this.audioQueue.forEach(item => {
      if (item.source) {
        try {
          item.source.stop();
          item.source.disconnect();
        } catch (e) {}
      }
    });
    this.audioQueue = [];
    this.isPlayingAudio = false;

    // Stop current audio
    if (this.currentAudioSource) {
      try {
        this.currentAudioSource.stop();
        this.currentAudioSource.disconnect();
      } catch (e) {}
      this.currentAudioSource = null;
    }

    // Stop avatar speaking
    if (window.stopSpeaking) {
      window.stopSpeaking();
    }

    // Disconnect audio processors
    try { this.processor?.disconnect(); } catch {}
    try { this.sourceNode?.disconnect(); } catch {}
    this.processor = null;
    this.sourceNode = null;

    // Stop media streams
    try {
      this.userStream?.getTracks().forEach(t => t.stop());
    } catch {}
    this.userStream = null;
    this.audioOnlyStream = null;

    // Close websockets
    try { this.eviWS?.close(); } catch {}
    try { this.exprWS?.close(); } catch {}
    this.eviWS = null;
    this.exprWS = null;

    // Clear timers
    if (this.faceTimer) {
      clearInterval(this.faceTimer);
      this.faceTimer = null;
    }
    if (this._keepTO) {
      clearInterval(this._keepTO);
      this._keepTO = null;
    }

    // Reset UI
    const camPreview = document.getElementById('camPreview');
    if (camPreview) {
      camPreview.innerHTML = '<span style="opacity:.6">Camera preview</span>';
    }
    
    this.speaking = false;
    this._lipRAF && cancelAnimationFrame(this._lipRAF);
    
    // Stop lipsync
    this.dispatchLipsync(0);
    
    this.btnStart.disabled = false;
    this.log('⏹ Система остановлена');
  }

  // ---------- Media Handling ----------
  async ensureMedia() {
    if (this.userStream) return;
    
    this.unlockAudio();
    
    const constraints = {
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      },
      video: {
        width: { ideal: 640 },
        height: { ideal: 360 }
      }
    };

    try {
      this.userStream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (e) {
      // Fallback to audio-only
      try {
        this.userStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        this.log('Video недоступно, используем только аудио: ' + e.message);
      } catch (audioError) {
        throw new Error('Не удается получить доступ к микрофону: ' + audioError.message);
      }
    }

    // Setup video preview if available
    if (this.userStream.getVideoTracks().length) {
      const video = document.createElement('video');
      video.muted = true;
      video.playsInline = true;
      video.srcObject = this.userStream;
      await video.play().catch(() => {});
      
      const camPreview = document.getElementById('camPreview');
      camPreview.innerHTML = '';
      camPreview.appendChild(video);
    }

    this.audioOnlyStream = new MediaStream(this.userStream.getAudioTracks());
  }

  streamMic() {
    if (!this.audioOnlyStream) return;
    
    this.log('🎤 Начинаем стрим микрофона (PCM16 16kHz WAV)');
    
    const ctx = this.audioCtx;
    this.sourceNode = ctx.createMediaStreamSource(this.audioOnlyStream);
    this.processor = ctx.createScriptProcessor(4096, 1, 1);
    this.analyser = ctx.createAnalyser();
    this.analyser.fftSize = 2048;

    this.sourceNode.connect(this.analyser);
    this.analyser.connect(this.processor);
    
    // Create silent sink to prevent feedback
    this._sinkGain = ctx.createGain();
    this._sinkGain.gain.value = 0.0;
    this.processor.connect(this._sinkGain);
    this._sinkGain.connect(ctx.destination);

    this.processor.onaudioprocess = (event) => {
      const channel = event.inputBuffer.getChannelData(0);
      const resampled = (event.inputBuffer.sampleRate === this.targetRate) ?
        channel : this.resample(channel, event.inputBuffer.sampleRate, this.targetRate);
      
      const wavB64 = this.pcm16WavBase64(resampled, this.targetRate);
      this.lastUserAudioTs = Date.now();

      // Send to EVI
      if (this.eviWS && this.eviWS.readyState === WebSocket.OPEN) {
        this.eviWS.send(JSON.stringify({
          type: 'audio_input',
          data: wavB64
        }));
      }

      // Send to Expressions (prosody)
      if (this.exprWS && this.exprWS.readyState === WebSocket.OPEN && this.exprCanStream) {
        this.exprWS.send(JSON.stringify({ data: wavB64 }));
      }
    };

    // Keep-alive для поддержания соединения
    this.startKeepAlive();
  }

  startKeepAlive() {
    this._keepTO && clearInterval(this._keepTO);
    this._keepTO = setInterval(() => {
      if (!this.eviWS || this.eviWS.readyState !== WebSocket.OPEN) return;
      if (Date.now() - this.lastUserAudioTs < 9000) return;
      
      // Send silence
      const silence = new Float32Array(160);
      const b64 = this.pcm16WavBase64(silence, this.targetRate);
      try {
        this.eviWS.send(JSON.stringify({
          type: 'audio_input',
          data: b64
        }));
      } catch (e) {
        console.warn('Keep-alive send failed:', e);
      }
    }, 3000);
  }

  streamFace() {
    if (!this.userStream.getVideoTracks().length) return;
    
    const video = document.querySelector('#camPreview video');
    if (!video) return;

    this.log('👤 Начинаем стрим лица');
    
    const W = 320, H = 240;
    const canvas = document.createElement('canvas');
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');

    this.faceTimer = setInterval(async () => {
      if (!this.exprWS || this.exprWS.readyState !== WebSocket.OPEN || !this.exprCanStream) {
        return;
      }

      try {
        ctx.drawImage(video, 0, 0, W, H);
        const blob = await new Promise(resolve => 
          canvas.toBlob(resolve, 'image/jpeg', 0.8)
        );
        
        if (!blob) return;
        
        const b64 = await this.blobToBase64(blob);
        this.exprWS.send(JSON.stringify({ data: b64 }));
        
      } catch (e) {
        console.error('Face stream error:', e);
      }
    }, 200);
  }

  // ---------- Hume WebSocket Connections ----------
  async openExpressions(key) {
    this.log('🔌 Подключаемся к Expressions API...');
    
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(
        'wss://api.hume.ai/v0/stream/models?api_key=' + encodeURIComponent(key)
      );
      
      this.exprWS = ws;
      this.exprCanStream = false;

      const timeout = setTimeout(() => {
        try { ws.close(); } catch {}
        reject(new Error('Expressions connection timeout'));
      }, 10000);

      ws.onopen = () => {
        clearTimeout(timeout);
        this.log('✅ Expressions подключено');
        
        // Configure models
        ws.send(JSON.stringify({
          type: 'models',
          models: { face: {}, prosody: {} }
        }));
        
        setTimeout(() => {
          this.exprCanStream = true;
        }, 600);
        
        resolve();
      };

      ws.onmessage = (event) => {
        let data;
        try {
          data = JSON.parse(event.data);
        } catch {
          return;
        }
        
        if (!this.exprCanStream) this.exprCanStream = true;
        this.handleExpressionsMessage(data);
      };

      ws.onerror = (error) => {
        clearTimeout(timeout);
        this.log('❌ Expressions socket error');
        console.error('Expressions error:', error);
      };

      ws.onclose = (event) => {
        this.log(`Expressions closed (code: ${event.code || 'n/a'})`);
        this.exprWS = null;
        if (this.faceTimer) {
          clearInterval(this.faceTimer);
          this.faceTimer = null;
        }
      };
    });
  }

  async openEVI(key, configId) {
    this.log('🔌 Подключаемся к EVI API...');
    
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(
        'wss://api.hume.ai/v0/evi/chat?api_key=' + encodeURIComponent(key) +
        '&config_id=' + encodeURIComponent(configId)
      );
      
      this.eviWS = ws;

      const timeout = setTimeout(() => {
        try { ws.close(); } catch {}
        reject(new Error('EVI connection timeout'));
      }, 10000);

      ws.onopen = () => {
        clearTimeout(timeout);
        this.log('✅ EVI подключено');
        
        // Configure session
        ws.send(JSON.stringify({
          type: 'session_settings',
          session_settings: {
            audio: {
              encoding: 'linear16',
              sample_rate_hz: this.targetRate
            }
          }
        }));
        
        resolve();
      };

      ws.onmessage = (event) => {
        let data;
        try {
          data = JSON.parse(event.data);
        } catch {
          return;
        }

        this.handleEVIMessage(data);
      };

      ws.onerror = (error) => {
        clearTimeout(timeout);
        this.log('❌ EVI socket error');
        console.error('EVI error:', error);
      };

      ws.onclose = (event) => {
        this.log(`EVI closed (code: ${event.code || 'n/a'})`);
        this.eviWS = null;
      };
    });
  }

  // ---------- Message Handlers ----------
  handleEVIMessage(data) {
    if (data?.type === 'audio_output' && data?.data) {
      this.queueAudio(data.data);
    }

    if (data?.type === 'assistant_message' && data?.message) {
      const content = data.message?.content || '...';
      this.log('🤖 EVI: ' + content.substring(0, 50) + (content.length > 50 ? '...' : ''));
    }

    if (data?.type === 'error') {
      this.log('❌ EVI error: ' + (data.message || 'Unknown error'));
    }
  }

  handleExpressionsMessage(data) {
    // Extract emotions from response
    const extractEmotions = (root) => {
      if (!root) return [];
      const arr = Array.isArray(root) ? root : (root.predictions || root.results || root.output || []);
      const item = Array.isArray(arr) ? arr[0] : arr;
      const emotions = item?.emotions || item?.scores || item?.probabilities || [];
      return Array.isArray(emotions) ? emotions : 
        Object.entries(emotions).map(([name, score]) => ({ name, score }));
    };

    const faceEmotions = extractEmotions(data.face);
    const prosodyEmotions = extractEmotions(data.prosody);

    // Extract face pose data
    if (data.face && data.face.predictions && data.face.predictions[0]) {
      const facePred = data.face.predictions[0];
      if (facePred.box) {
        this.lastFacePose = {
          facePose: {
            yaw: facePred.box.yaw || 0,
            pitch: facePred.box.pitch || 0,
            roll: facePred.box.roll || 0
          }
        };
      }
    }

    // Combine face and prosody emotions
    const WF = 0.6, WV = 0.4; // Weights for face and voice
    
    const pickEmotionScore = (keywords, emotions) => {
      let score = 0;
      for (const emotion of emotions) {
        const name = (emotion.name || '').toLowerCase();
        if (keywords.some(keyword => name.includes(keyword))) {
          score += (emotion.score ?? emotion.probability ?? 0);
        }
      }
      return Math.max(0, Math.min(1, score));
    };

    // Update emotion values with smoothing
    let emotionUpdated = false;
    for (const emotionKey of Object.keys(this.EMOS)) {
      const keywords = this.EMOS[emotionKey];
      const combinedValue = WF * pickEmotionScore(keywords, faceEmotions) + 
                           WV * pickEmotionScore(keywords, prosodyEmotions);
      
      if (combinedValue > 0.1) {
        console.log(`[Emotions] ${emotionKey}: ${combinedValue.toFixed(3)}`);
        emotionUpdated = true;
      }
      
      this.setEmotion(emotionKey, combinedValue);
    }

    if (emotionUpdated) {
      console.log('[Emotions] Updating avatar with new emotions');
    }

    this.updateHUDTick();
  }

  // ---------- Emotion System ----------
  setEmotion(name, value) {
    const previous = this.emoValues[name] || 0;
    const smoothed = previous + this.EMA_ALPHA * (Math.max(0, Math.min(1, value || 0)) - previous);
    this.emoValues[name] = smoothed;
    this.renderEmotionHUD();
    this.updateAvatar();
  }

  updateHUDTick() {
    const dot = document.getElementById('hudTick');
    if (!dot) return;
    
    dot.style.background = '#3bd67f';
    clearTimeout(this._hudDotTO);
    this._hudDotTO = setTimeout(() => {
      dot.style.background = '#444';
    }, 600);
  }

  renderEmotionHUD() {
    for (const [emotionKey, value] of Object.entries(this.emoValues)) {
      const bar = document.getElementById('bar-' + emotionKey);
      const val = document.getElementById('val-' + emotionKey);
      const percentage = Math.round(value * 100);
      
      if (bar) bar.style.width = percentage + '%';
      if (val) val.textContent = percentage + '%';
    }
  }

  updateAvatar() {
    // Find the dominant emotion
    const dominantEmotion = Object.entries(this.emoValues)
      .sort((a, b) => b[1] - a[1])[0];

    console.log('[Emotions] Current values:', this.emoValues);
    console.log('[Emotions] Dominant emotion:', dominantEmotion);

    if (dominantEmotion && dominantEmotion[1] > 0.15) {
      console.log('[Avatar] Applying emotion:', dominantEmotion[0], 'intensity:', dominantEmotion[1]);

      if (window.applyEmotion) {
        window.applyEmotion(dominantEmotion[0], dominantEmotion[1]);
      } else {
        // Queue emotion update for when avatar is ready
        window.addEventListener('avatar:ready', () => {
          if (window.applyEmotion) {
            window.applyEmotion(dominantEmotion[0], dominantEmotion[1]);
          }
        }, { once: true });
        console.warn('[Avatar] TalkingHead not ready, queued emotion update');
      }
    } else {
      console.log('[Avatar] Setting neutral state');
      if (window.applyEmotion) {
        window.applyEmotion('neutral', 0.5);
      }
    }

    // Also send face pose data if available
    if (this.lastFacePose) {
      this.dispatchExpression(this.lastFacePose);
    }
  }

  // ---------- Audio Playback System ----------
  queueAudio(base64AudioData) {
    this.audioQueue.push({
      data: base64AudioData,
      id: Date.now()
    });
    
    if (!this.isPlayingAudio) {
      this.processAudioQueue();
    }
  }

  async processAudioQueue() {
    if (this.audioQueue.length === 0) {
      this.isPlayingAudio = false;
      return;
    }

    this.isPlayingAudio = true;
    const audioItem = this.audioQueue.shift();

    try {
      await this.playAudioBase64(audioItem.data);
      
      // Small pause between audio fragments
      await new Promise(resolve => setTimeout(resolve, 100));
      
    } catch (error) {
      this.log('Audio playback error: ' + error.message);
    }

    // Continue processing queue
    this.processAudioQueue();
  }

  async playAudioBase64(base64Data) {
    return new Promise(async (resolve, reject) => {
      try {
        this.unlockAudio();

        // Stop previous audio if playing
        if (this.currentAudioSource) {
          try {
            this.currentAudioSource.stop();
            this.currentAudioSource.disconnect();
          } catch (e) {}
          this.currentAudioSource = null;
        }

        // Stop previous lipsync animation
        if (this._lipRAF) {
          cancelAnimationFrame(this._lipRAF);
          this._lipRAF = 0;
        }
        this.speaking = false;

        // Try TalkingHead integration first
        if (this.avatarReady && window.talkingHead && window.playAudioWithLipsync) {
          console.log('[Audio] Using TalkingHead for audio playback');
          
          try {
            // Convert Base64 to ArrayBuffer
            const binaryString = atob(base64Data);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
              bytes[i] = binaryString.charCodeAt(i);
            }
            
            // Decode audio
            const audioBuffer = await this.audioCtx.decodeAudioData(bytes.buffer.slice());
            
            // Use TalkingHead's audio system
            const audioObject = {
              audio: audioBuffer,
              words: [], // Could add words if available from EVI
              wtimes: [],
              wdurations: []
            };
            
            window.talkingHead.speakAudio(audioObject, { lipsyncLang: 'en' });
            
            // Resolve after estimated duration
            const duration = audioBuffer.duration * 1000;
            setTimeout(() => resolve(), duration);
            
            return; // Exit early if TalkingHead worked
            
          } catch (talkingHeadError) {
            console.warn('[Audio] TalkingHead playback failed, using fallback:', talkingHeadError);
          }
        }

        // Fallback to Web Audio API
        await this.playAudioFallback(base64Data, resolve, reject);

      } catch (error) {
        this.log('Audio play error: ' + error.message);
        this.currentAudioSource = null;
        this.speaking = false;
        reject(error);
      }
    });
  }

  async playAudioFallback(base64Data, resolve, reject) {
    try {
      // Decode Base64 to audio buffer
      const binaryString = atob(base64Data);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      const audioBuffer = await this.audioCtx.decodeAudioData(bytes.buffer.slice());

      // Create audio source
      const source = this.audioCtx.createBufferSource();
      source.buffer = audioBuffer;

      const gain = this.audioCtx.createGain();
      gain.gain.value = 0.7;
      
      const analyser = this.audioCtx.createAnalyser();
      analyser.fftSize = 512;

      source.connect(gain).connect(analyser).connect(this.audioCtx.destination);

      // Store reference BEFORE starting
      this.currentAudioSource = source;

      // Set up lipsync animation
      this.speaking = true;
      const audioData = new Uint8Array(analyser.frequencyBinCount);
      
      const lipsyncLoop = () => {
        if (!this.speaking || !this.currentAudioSource) return;
        
        analyser.getByteTimeDomainData(audioData);
        
        let sum = 0;
        for (let i = 0; i < audioData.length; i++) {
          const sample = (audioData[i] - 128) / 128;
          sum += sample * sample;
        }
        
        const rms = Math.sqrt(sum / audioData.length);
        const level = Math.min(1, Math.max(0, rms * 8.0));

        // Dispatch lipsync event
        this.dispatchLipsync(level);

        this._lipRAF = requestAnimationFrame(lipsyncLoop);
      };

      // Cleanup function
      const cleanup = () => {
        this.speaking = false;
        this._lipRAF && cancelAnimationFrame(this._lipRAF);
        this.currentAudioSource = null;
        this.dispatchLipsync(0); // Stop lipsync
        resolve();
      };

      // Set up event handlers
      source.onended = cleanup;

      // Fallback timeout
      const duration = Math.max(300, audioBuffer.duration * 1000 + 100);
      clearTimeout(this._speakTO);
      this._speakTO = setTimeout(cleanup, duration + 200);

      // Start audio and lipsync
      source.start();
      setTimeout(() => {
        this._lipRAF = requestAnimationFrame(lipsyncLoop);
      }, 100);

    } catch (error) {
      reject(error);
    }
  }

  // ---------- Event Dispatchers ----------
  dispatchLipsync(level) {
    window.dispatchEvent(new CustomEvent('octo:lipsync', {
      detail: { level: level }
    }));
  }

  dispatchExpression(data) {
    window.dispatchEvent(new CustomEvent('octo:expression', {
      detail: data
    }));
  }

  // ---------- Audio Processing Utilities ----------
  resample(inputBuffer, srcRate, dstRate) {
    const ratio = dstRate / srcRate;
    const length = Math.round(inputBuffer.length * ratio);
    const result = new Float32Array(length);
    
    for (let i = 0; i < length; i++) {
      const srcIndex = i / ratio;
      const index0 = Math.floor(srcIndex);
      const index1 = Math.min(index0 + 1, inputBuffer.length - 1);
      const fraction = srcIndex - index0;
      
      result[i] = inputBuffer[index0] * (1 - fraction) + inputBuffer[index1] * fraction;
    }
    
    return result;
  }

  pcm16WavBase64(floatArray, sampleRate) {
    const samples = floatArray;
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const bytes = new Uint8Array(buffer);

    // WAV file header
    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        bytes[offset + i] = string.charCodeAt(i);
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true);

    // Convert float samples to 16-bit PCM
    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
      let sample = Math.max(-1, Math.min(1, samples[i]));
      sample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
      view.setInt16(offset, sample, true);
      offset += 2;
    }

    return btoa(String.fromCharCode(...bytes));
  }

  async blobToBase64(blob) {
    const arrayBuffer = await blob.arrayBuffer();
    const bytes = new Uint8Array(arrayBuffer);
    let binaryString = '';
    
    for (let i = 0; i < bytes.length; i++) {
      binaryString += String.fromCharCode(bytes[i]);
    }
    
    return binaryString ? btoa(binaryString) : '';
  }
} // End of OctoApp class

// Initialize application when DOM is ready
window.addEventListener('DOMContentLoaded', () => {
  console.log('[OctoApp] Starting initialization...');
  window.OCTO = new OctoApp();
  console.log('[OctoApp] Initialized successfully');

  // Debug information
  console.log('[OctoApp] Checking avatar system availability:', typeof window.applyEmotion);
  console.log('[OctoApp] Log element found:', document.getElementById('log') !== null);

  // Test emotion after 2 seconds if avatar is ready
  setTimeout(() => {
    if (window.applyEmotion) {
      console.log('[OctoApp] Testing emotion: joy');
      window.applyEmotion('joy', 0.5);
    }
  }, 2000);
});

// Global error handler
window.addEventListener('error', (event) => {
  console.error('[OctoApp] Global error:', event);
  try {
    window.OCTO?.log('JS error: ' + event.message);
  } catch (e) {
    console.error('Could not log error:', e);
  }
});

})(); // End of IIFE