// main.js - Ready Player Me GLB support (without VRM)

let scene, camera, renderer, currentModel, mixer, clock;
let idleTimer = 0;
let lastEmotion = null;
let blinkTimer = 0;
let blinkInterval = Math.random() * 4 + 2;
let morphTargets = {};

window.applyEmotion = function(emotionName, intensity = 1.0) {
  if (!currentModel || !morphTargets) return;
  
  // Ready Player Me использует другие названия морф-таргетов
  const map = {
    joy: ['mouthSmile', 'mouthSmileLeft', 'mouthSmileRight'],
    surprise: ['mouthOpen', 'eyeWideLeft', 'eyeWideRight'],
    anger: ['mouthFrownLeft', 'mouthFrownRight', 'browDownLeft', 'browDownRight'],
    sadness: ['mouthFrownLeft', 'mouthFrownRight'],
    neutral: []
  };

  // Сбрасываем все морфы
  Object.keys(morphTargets).forEach(key => {
    if (morphTargets[key]) morphTargets[key].value = 0;
  });

  // Применяем новые
  const morphNames = map[emotionName] || [];
  morphNames.forEach(name => {
    if (morphTargets[name]) {
      morphTargets[name].value = intensity;
    }
  });

  lastEmotion = emotionName;
  idleTimer = 0;
};

window.addEventListener('octo:lipsync', (e) => {
  if (!morphTargets) return;
  const level = e.detail.level || 0;
  
  // Липсинк через морф-таргеты Ready Player Me
  if (morphTargets.mouthOpen) {
    morphTargets.mouthOpen.value = level;
  }
  if (morphTargets.viseme_aa) {
    morphTargets.viseme_aa.value = level;
  }
});

window.addEventListener('octo:expression', (e) => {
  const data = e.detail;
  if (!currentModel) return;
  
  // Ищем голову в иерархии
  let head = null;
  currentModel.traverse((child) => {
    if (child.name && (child.name.toLowerCase().includes('head') || child.name === 'Head')) {
      head = child;
    }
  });
  
  if (!head) return;

  let x = 0, y = 0;
  if (data.facePose) {
    x = Math.max(-1, Math.min(1, (data.facePose.yaw || 0) / 30));
    y = Math.max(-1, Math.min(1, -(data.facePose.pitch || 0) / 30));
  } else if (data.userPosition) {
    x = data.userPosition.x || 0;
    y = -data.userPosition.y || 0;
  }

  head.rotation.y = x * 0.5;
  head.rotation.x = y * 0.5;
});

function triggerBlink() {
  if (!morphTargets) return;
  
  // Ready Player Me blink морф-таргеты
  const blinkMorphs = ['eyeBlinkLeft', 'eyeBlinkRight', 'eyesClosed'];
  
  blinkMorphs.forEach(name => {
    if (morphTargets[name]) {
      morphTargets[name].value = 1;
      setTimeout(() => {
        if (morphTargets[name]) morphTargets[name].value = 0;
      }, 100);
    }
  });
  
  blinkInterval = Math.random() * 4 + 2;
  blinkTimer = 0;
}

function findMorphTargets(model) {
  morphTargets = {};
  
  model.traverse((child) => {
    if (child.isMesh && child.morphTargetInfluences && child.morphTargetDictionary) {
      console.log('Found morph targets:', Object.keys(child.morphTargetDictionary));
      
      // Создаем объекты для управления морф-таргетами
      Object.keys(child.morphTargetDictionary).forEach(key => {
        const index = child.morphTargetDictionary[key];
        morphTargets[key] = {
          get value() { return child.morphTargetInfluences[index]; },
          set value(v) { child.morphTargetInfluences[index] = v; }
        };
      });
    }
  });
  
  return morphTargets;
}

async function init() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0a0a); // Темный фон
  
  camera = new THREE.PerspectiveCamera(35, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.set(0, 1.3, 3); // Камера чуть ниже
  camera.lookAt(0, 1.0, 0); // Смотрим на центр торса

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.shadowMap.enabled = true;
  document.getElementById("avatarContainer").appendChild(renderer.domElement);

  // Освещение для темного фона
  const ambientLight = new THREE.AmbientLight(0x404040, 1.2); // Приглушенный свет
  scene.add(ambientLight);
  
  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight.position.set(0.5, 1, 0.5);
  directionalLight.castShadow = true;
  scene.add(directionalLight);
  
  // Дополнительный свет спереди для лица
  const frontLight = new THREE.DirectionalLight(0xffffff, 0.3);
  frontLight.position.set(0, 1, 2);
  scene.add(frontLight);

  clock = new THREE.Clock();

  const loader = new THREE.GLTFLoader();
  
  // Ready Player Me URL или локальный файл
  const avatarUrl = 'https://octopuzzler.netlify.app/avatar.glb';

  loader.load(
    avatarUrl,
    (gltf) => {
      console.log('GLTF loaded successfully:', gltf);
      
      const model = gltf.scene;
      
      // Настройка теней
      model.traverse((child) => {
        if (child.isMesh) {
          child.castShadow = true;
          child.receiveShadow = true;
          
          // Логируем имена всех объектов для отладки
          if (child.name) {
            console.log('Mesh found:', child.name);
          }
        }
      });
      
      // Центрируем и позиционируем модель правильно
      const box = new THREE.Box3().setFromObject(model);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      
      model.position.x = -center.x;
      model.position.y = 0; // Ставим на уровень 0
      model.position.z = -center.z;
      
      // Масштабируем до нужной высоты
      const targetHeight = 1.8;
      if (size.y !== targetHeight) {
        model.scale.multiplyScalar(targetHeight / size.y);
      }
      
      // Корректируем позицию после масштабирования
      const newBox = new THREE.Box3().setFromObject(model);
      model.position.y = -newBox.min.y; // Поднимаем так, чтобы ноги были на уровне 0
      
      scene.add(model);
      currentModel = model;
      
      // Находим морф-таргеты
      findMorphTargets(model);
      
      // Настройка анимаций если есть
      if (gltf.animations && gltf.animations.length > 0) {
        console.log('Animations found:', gltf.animations.length);
        mixer = new THREE.AnimationMixer(model);
        
        // Запускаем первую анимацию (обычно idle)
        const action = mixer.clipAction(gltf.animations[0]);
        action.play();
      }
      
      animate();
    },
    (progress) => {
      if (progress.total) {
        const percent = (progress.loaded / progress.total * 100).toFixed(0);
        console.log('Loading:', percent + '%');
      }
    },
    (error) => {
      console.error('Error loading avatar:', error);
      
      // Создаем простой куб как fallback
      const geometry = new THREE.BoxGeometry(0.5, 1.5, 0.3);
      const material = new THREE.MeshStandardMaterial({ 
        color: 0x4444ff,
        metalness: 0.3,
        roughness: 0.7
      });
      const cube = new THREE.Mesh(geometry, material);
      cube.position.y = 0.75;
      cube.castShadow = true;
      scene.add(cube);
      currentModel = cube;
      animate();
    }
  );
}

function animate() {
  requestAnimationFrame(animate);
  
  const delta = clock.getDelta();
  
  // Обновляем анимации
  if (mixer) {
    mixer.update(delta);
  }
  
  // Idle анимация - легкое покачивание
  if (currentModel) {
    currentModel.rotation.y = Math.sin(Date.now() * 0.0005) * 0.05;
  }
  
  // Таймеры для эмоций и морганий
  idleTimer += delta;
  blinkTimer += delta;

  if (idleTimer > 5 && lastEmotion !== 'neutral') {
    applyEmotion('neutral');
    lastEmotion = 'neutral';
  }
  
  if (blinkTimer > blinkInterval) {
    triggerBlink();
  }
  
  renderer.render(scene, camera);
}

// Обработка изменения размера окна
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

window.addEventListener('DOMContentLoaded', init);