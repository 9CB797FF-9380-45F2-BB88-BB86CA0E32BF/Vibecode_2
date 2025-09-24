// AI 실시간 객체 칼로리 추정 서비스 - 메인 JavaScript 파일

class ObjectDetectionModel {
    constructor() {
        this.model = null;
        this.isLoaded = false;
        this.isLoading = false;
        this.loadingProgress = 0;
        this.onStatusUpdate = null;
    }
    
    async loadModel() {
        if (this.isLoaded || this.isLoading) {
            return this.model;
        }
        
        this.isLoading = true;
        this.loadingProgress = 0;
        this.showModelLoadingUI();
        this.updateStatus('AI 모델 로딩 중...', 'loading');
        this.updateModelLoadingProgress(0, 'AI 모델 로딩 준비 중...');
        
        try {
            // COCO-SSD 모델 로드 (TensorFlow.js AutoML 사용)
            this.updateStatus('COCO-SSD 모델 다운로드 중...', 'loading');
            this.loadingProgress = 30;
            this.updateModelLoadingProgress(30, 'COCO-SSD 모델 다운로드 중...');
            
            // TensorFlow.js AutoML을 사용하여 COCO-SSD 모델 로드
            this.model = await tf.automl.loadObjectDetection('https://tfhub.dev/tensorflow/tfjs-model/ssd_mobilenet_v2/1/default/1');
            
            this.loadingProgress = 80;
            this.updateStatus('모델 초기화 중...', 'loading');
            this.updateModelLoadingProgress(80, '모델 초기화 중...');
            
            // 모델 로드 완료
            this.isLoaded = true;
            this.isLoading = false;
            this.loadingProgress = 100;
            this.updateStatus('AI 모델 로드 완료!', 'success');
            this.updateModelLoadingProgress(100, 'AI 모델 로드 완료!');
            
            // UI 숨기기 (약간의 지연 후)
            setTimeout(() => {
                this.hideModelLoadingUI();
            }, 2000);
            
            console.log('Object detection model loaded successfully');
            return this.model;
            
        } catch (error) {
            this.isLoading = false;
            this.isLoaded = false;
            this.loadingProgress = 0;
            
            console.error('Failed to load object detection model:', error);
            this.updateStatus('AI 모델 로드 실패: ' + error.message, 'error');
            this.updateModelLoadingProgress(0, 'AI 모델 로드 실패: ' + error.message);
            
            // 폴백: 다른 모델 시도
            try {
                this.updateStatus('대체 모델 로드 시도 중...', 'loading');
                this.updateModelLoadingProgress(10, '대체 모델 로드 시도 중...');
                this.model = await tf.automl.loadObjectDetection('https://tfhub.dev/google/tfjs-model/ssd_mobilenet_v2/1/default/1');
                this.isLoaded = true;
                this.isLoading = false;
                this.loadingProgress = 100;
                this.updateStatus('대체 AI 모델 로드 완료!', 'success');
                this.updateModelLoadingProgress(100, '대체 AI 모델 로드 완료!');
                
                setTimeout(() => {
                    this.hideModelLoadingUI();
                }, 2000);
                
                return this.model;
            } catch (fallbackError) {
                console.error('Fallback model also failed:', fallbackError);
                this.updateStatus('모든 AI 모델 로드 실패', 'error');
                this.updateModelLoadingProgress(0, '모든 AI 모델 로드 실패');
                throw fallbackError;
            }
        }
    }
    
    async detectObjects(imageElement, options = {}) {
        if (!this.isLoaded || !this.model) {
            throw new Error('Model not loaded. Call loadModel() first.');
        }
        
        try {
            const defaultOptions = {
                score: 0.5,    // 최소 신뢰도 점수
                iou: 0.5,      // Intersection over Union 임계값
                topk: 20       // 최대 탐지 객체 수
            };
            
            const detectionOptions = { ...defaultOptions, ...options };
            const predictions = await this.model.detect(imageElement, detectionOptions);
            
            return predictions;
        } catch (error) {
            console.error('Object detection failed:', error);
            throw error;
        }
    }
    
    updateStatus(message, type) {
        if (this.onStatusUpdate) {
            this.onStatusUpdate(message, type, this.loadingProgress);
        }
    }
    
    showModelLoadingUI() {
        const loadingSection = document.getElementById('modelLoadingSection');
        if (loadingSection) {
            loadingSection.style.display = 'block';
        }
    }
    
    hideModelLoadingUI() {
        const loadingSection = document.getElementById('modelLoadingSection');
        if (loadingSection) {
            loadingSection.style.display = 'none';
        }
    }
    
    updateModelLoadingProgress(progress, statusText) {
        const progressBar = document.getElementById('modelProgressBar');
        const progressText = document.getElementById('modelProgressText');
        const statusTextElement = document.getElementById('modelStatusText');
        
        if (progressBar) {
            progressBar.style.width = progress + '%';
        }
        
        if (progressText) {
            progressText.textContent = Math.round(progress) + '%';
        }
        
        if (statusTextElement && statusText) {
            statusTextElement.textContent = statusText;
        }
    }
    
    setStatusCallback(callback) {
        this.onStatusUpdate = callback;
    }
    
    getModelInfo() {
        return {
            isLoaded: this.isLoaded,
            isLoading: this.isLoading,
            progress: this.loadingProgress,
            modelType: 'COCO-SSD'
        };
    }
    
    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
        this.isLoaded = false;
        this.isLoading = false;
        this.loadingProgress = 0;
    }
}

class ObjectDetectionPipeline {
    constructor(videoElement, objectDetectionModel) {
        this.video = videoElement;
        this.model = objectDetectionModel;
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        
        // Detection settings
        this.isDetecting = false;
        this.detectionInterval = 100; // ms between detections
        this.confidenceThreshold = 0.5;
        this.maxDetections = 20;
        
        // Performance tracking
        this.lastDetectionTime = 0;
        this.detectionCount = 0;
        this.fps = 0;
        this.lastFpsUpdate = 0;
        
        // Detection results
        this.currentDetections = [];
        this.onDetectionUpdate = null;
        
        // Animation frame
        this.animationId = null;
    }
    
    startDetection() {
        if (this.isDetecting) {
            return;
        }
        
        this.isDetecting = true;
        this.detectionCount = 0;
        this.lastDetectionTime = 0;
        this.lastFpsUpdate = performance.now();
        
        console.log('Object detection pipeline started');
        this.detectLoop();
    }
    
    stopDetection() {
        this.isDetecting = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        
        // Clear canvas
        this.clearCanvas();
        this.currentDetections = [];
        
        console.log('Object detection pipeline stopped');
    }
    
    detectLoop() {
        if (!this.isDetecting) {
            return;
        }
        
        const now = performance.now();
        
        // Update FPS counter
        if (now - this.lastFpsUpdate >= 1000) {
            this.fps = this.detectionCount;
            this.detectionCount = 0;
            this.lastFpsUpdate = now;
        }
        
        // Check if it's time for next detection
        if (now - this.lastDetectionTime >= this.detectionInterval) {
            this.performDetection();
            this.lastDetectionTime = now;
            this.detectionCount++;
        }
        
        // Continue the loop
        this.animationId = requestAnimationFrame(() => this.detectLoop());
    }
    
    async performDetection() {
        if (!this.model || !this.model.isLoaded || !this.video.videoWidth) {
            return;
        }
        
        try {
            // Perform object detection
            const detections = await this.model.detectObjects(this.video, {
                score: this.confidenceThreshold,
                topk: this.maxDetections
            });
            
            // Process and filter results
            this.currentDetections = this.processDetections(detections);
            
            // Update visualization
            this.updateVisualization();
            
            // Notify listeners
            if (this.onDetectionUpdate) {
                this.onDetectionUpdate(this.currentDetections, this.fps);
            }
            
        } catch (error) {
            console.error('Detection error:', error);
        }
    }
    
    processDetections(detections) {
        return detections
            .filter(detection => detection.score >= this.confidenceThreshold)
            .map(detection => ({
                class: detection.label,
                confidence: Math.round(detection.score * 100) / 100,
                bbox: {
                    x: Math.round(detection.box.left),
                    y: Math.round(detection.box.top),
                    width: Math.round(detection.box.width),
                    height: Math.round(detection.box.height)
                }
            }))
            .sort((a, b) => b.confidence - a.confidence);
    }
    
    updateVisualization() {
        // Clear previous drawings
        this.clearCanvas();
        
        // Set canvas size to match video
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        
        // Draw bounding boxes and labels
        this.currentDetections.forEach((detection, index) => {
            this.drawBoundingBox(detection, index);
        });
    }
    
    drawBoundingBox(detection, index) {
        const { x, y, width, height } = detection.bbox;
        
        // Choose color based on detection index
        const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff'];
        const color = colors[index % colors.length];
        
        // Draw bounding box
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 3;
        this.ctx.strokeRect(x, y, width, height);
        
        // Draw label background
        const label = `${detection.class} (${Math.round(detection.confidence * 100)}%)`;
        const labelWidth = this.ctx.measureText(label).width + 10;
        const labelHeight = 20;
        
        this.ctx.fillStyle = color;
        this.ctx.fillRect(x, y - labelHeight, labelWidth, labelHeight);
        
        // Draw label text
        this.ctx.fillStyle = 'white';
        this.ctx.font = '14px Arial';
        this.ctx.fillText(label, x + 5, y - 5);
    }
    
    clearCanvas() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    setDetectionInterval(interval) {
        this.detectionInterval = Math.max(50, interval); // Minimum 50ms
    }
    
    setConfidenceThreshold(threshold) {
        this.confidenceThreshold = Math.max(0.1, Math.min(1.0, threshold));
    }
    
    setMaxDetections(max) {
        this.maxDetections = Math.max(1, Math.min(50, max));
    }
    
    setDetectionCallback(callback) {
        this.onDetectionUpdate = callback;
    }
    
    getDetectionStats() {
        return {
            isDetecting: this.isDetecting,
            fps: this.fps,
            detectionCount: this.detectionCount,
            currentDetections: this.currentDetections.length,
            confidenceThreshold: this.confidenceThreshold,
            detectionInterval: this.detectionInterval
        };
    }
    
    dispose() {
        this.stopDetection();
        this.clearCanvas();
        this.currentDetections = [];
        this.onDetectionUpdate = null;
    }
}

class CalibrationController {
    constructor() {
        // Known dimensions of reference objects in millimeters
        this.REFERENCE_OBJECTS = {
            'credit card': { width: 85.6, height: 53.98, area: 85.6 * 53.98 }
            // Future objects can be added here, e.g., A4 paper, coins
        };

        this.mmPerPixel = null;
        this.lastCalibrationTime = 0;
        this.isCalibrated = false;
        this.calibrationObject = null;
    }

    // Attempt to calibrate using the detected objects
    update(detections) {
        for (const detection of detections) {
            if (this.REFERENCE_OBJECTS[detection.class]) {
                const refObject = this.REFERENCE_OBJECTS[detection.class];
                const detectedWidth = detection.bbox.width;
                const detectedHeight = detection.bbox.height;

                // Calculate mm/pixel ratio from both width and height and average them
                const mmPerPixelWidth = refObject.width / detectedWidth;
                const mmPerPixelHeight = refObject.height / detectedHeight;
                
                // A simple average. More complex logic could be used here.
                const newMmPerPixel = (mmPerPixelWidth + mmPerPixelHeight) / 2;

                // Simple smoothing to stabilize the value
                if (this.mmPerPixel) {
                    this.mmPerPixel = this.mmPerPixel * 0.9 + newMmPerPixel * 0.1;
                } else {
                    this.mmPerPixel = newMmPerPixel;
                }

                this.isCalibrated = true;
                this.lastCalibrationTime = performance.now();
                this.calibrationObject = detection.class;
                
                // Once a reference object is found and used, we can stop for this frame.
                return true;
            }
        }
        return false;
    }

    // Get the current calibration status
    getStatus() {
        return {
            isCalibrated: this.isCalibrated,
            mmPerPixel: this.mmPerPixel,
            lastCalibrationTime: this.lastCalibrationTime,
            calibrationObject: this.calibrationObject
        };
    }

    // Reset calibration if the object is lost for a while or on demand
    reset() {
        this.mmPerPixel = null;
        this.isCalibrated = false;
        this.calibrationObject = null;
    }
}

class VolumeEstimator {
    constructor() {
        // Pre-defined database of common objects and their typical dimensions (in mm)
        // or volume (in cm^3). This is a simplified example.
        this.OBJECT_DATABASE = {
            'cup': { shape: 'cylinder', avg_diameter: 75, avg_height: 95 },
            'bottle': { shape: 'cylinder', avg_diameter: 65, avg_height: 230 },
            'apple': { shape: 'sphere', avg_diameter: 80 },
            'orange': { shape: 'sphere', avg_diameter: 75 },
            'banana': { shape: 'cylinder', avg_diameter: 35, avg_height: 180 }, // Approximated as a cylinder
            'bowl': { shape: 'hemisphere', avg_diameter: 150 },
            // Add more objects as needed
        };
    }

    estimate(detection, calibrationStatus) {
        if (!calibrationStatus.isCalibrated || !this.OBJECT_DATABASE[detection.class]) {
            return {
                volume: null,
                realWidth: null,
                realHeight: null,
                error: !calibrationStatus.isCalibrated ? "Not calibrated" : "Object not in database"
            };
        }

        const objectInfo = this.OBJECT_DATABASE[detection.class];
        const mmPerPixel = calibrationStatus.mmPerPixel;
        
        const realWidth = detection.bbox.width * mmPerPixel;
        const realHeight = detection.bbox.height * mmPerPixel;

        let volume = 0; // Volume in cm^3

        switch (objectInfo.shape) {
            case 'sphere':
                // V = 4/3 * pi * r^3
                const radiusSphere = (realWidth / 2) / 10; // Convert mm to cm
                volume = (4 / 3) * Math.PI * Math.pow(radiusSphere, 3);
                break;
            case 'cylinder':
                // V = pi * r^2 * h
                const radiusCylinder = (realWidth / 2) / 10; // Convert mm to cm
                const heightCylinder = realHeight / 10; // Convert mm to cm
                volume = Math.PI * Math.pow(radiusCylinder, 2) * heightCylinder;
                break;
            case 'hemisphere':
                 // V = 2/3 * pi * r^3
                 const radiusHemisphere = (realWidth / 2) / 10; // Convert mm to cm
                 volume = (2 / 3) * Math.PI * Math.pow(radiusHemisphere, 3);
                 break;
            default:
                // Fallback for undefined shapes: rough box volume
                // This is a very rough approximation.
                const depth = (realWidth + realHeight) / 2; // Rough depth estimate
                volume = (realWidth / 10) * (realHeight / 10) * (depth / 10);
                break;
        }
        
        return {
            volume: parseFloat(volume.toFixed(2)),
            realWidth: parseFloat(realWidth.toFixed(2)),
            realHeight: parseFloat(realHeight.toFixed(2)),
            error: null
        };
    }
}

class CameraController {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.startButton = document.getElementById('startButton');
        this.stopButton = document.getElementById('stopButton');
        this.statusMessage = document.getElementById('statusMessage');
        
        this.stream = null;
        this.isStreaming = false;
        
        // AI 모델 인스턴스 생성
        this.objectDetectionModel = new ObjectDetectionModel();
        
        // 객체 탐지 파이프라인 인스턴스 생성
        this.detectionPipeline = new ObjectDetectionPipeline(this.video, this.objectDetectionModel);
        
        // 보정 컨트롤러 인스턴스 생성
        this.calibrationController = new CalibrationController();

        // 부피 추정기 인스턴스 생성
        this.volumeEstimator = new VolumeEstimator();

        // 탐지 컨트롤 요소들
        this.detectionControls = document.getElementById('detectionControls');
        this.confidenceSlider = document.getElementById('confidenceSlider');
        this.confidenceValue = document.getElementById('confidenceValue');
        this.detectionIntervalSlider = document.getElementById('detectionIntervalSlider');
        this.detectionIntervalValue = document.getElementById('detectionIntervalValue');
        this.maxDetectionsSlider = document.getElementById('maxDetectionsSlider');
        this.maxDetectionsValue = document.getElementById('maxDetectionsValue');
        this.toggleDetectionButton = document.getElementById('toggleDetectionButton');
        this.detectionFps = document.getElementById('detectionFps');
        this.detectedObjects = document.getElementById('detectedObjects');

        // 보정 상태 UI 요소
        this.calibrationStatus = document.getElementById('calibrationStatus');
        this.calibrationValue = document.getElementById('calibrationValue');

        // 결과 표시 UI 요소
        this.objectList = document.getElementById('objectList');
        this.totalVolume = document.getElementById('totalVolume');
        
        this.initializeEventListeners();
        this.initializeDetectionControls();
        this.updateStatus('카메라를 시작하려면 버튼을 클릭하세요', 'default');
    }
    
    initializeEventListeners() {
        this.startButton.addEventListener('click', () => this.startCamera());
        this.stopButton.addEventListener('click', () => this.stopCamera());
        
        // 비디오 메타데이터 로드 이벤트
        this.video.addEventListener('loadedmetadata', () => {
            this.video.play();
            this.isStreaming = true;
            this.updateStatus('카메라가 성공적으로 시작되었습니다', 'success');
        });
        
        // 비디오 에러 이벤트
        this.video.addEventListener('error', (e) => {
            console.error('비디오 에러:', e);
            this.updateStatus('비디오 스트림에 오류가 발생했습니다', 'error');
        });
    }
    
    initializeDetectionControls() {
        // 신뢰도 슬라이더
        this.confidenceSlider.addEventListener('input', (e) => {
            const value = e.target.value;
            this.confidenceValue.textContent = value;
            this.detectionPipeline.setConfidenceThreshold(value / 100);
        });
        
        // 탐지 주기 슬라이더
        this.detectionIntervalSlider.addEventListener('input', (e) => {
            const value = e.target.value;
            this.detectionIntervalValue.textContent = value;
            this.detectionPipeline.setDetectionInterval(parseInt(value));
        });
        
        // 최대 탐지 수 슬라이더
        this.maxDetectionsSlider.addEventListener('input', (e) => {
            const value = e.target.value;
            this.maxDetectionsValue.textContent = value;
            this.detectionPipeline.setMaxDetections(parseInt(value));
        });
        
        // 탐지 토글 버튼
        this.toggleDetectionButton.addEventListener('click', () => {
            if (this.detectionPipeline.isDetecting) {
                this.detectionPipeline.stopDetection();
                this.toggleDetectionButton.textContent = '탐지 재시작';
                this.toggleDetectionButton.classList.add('paused');
            } else {
                this.detectionPipeline.startDetection();
                this.toggleDetectionButton.textContent = '탐지 일시정지';
                this.toggleDetectionButton.classList.remove('paused');
            }
        });
        
        // 탐지 결과 콜백 설정
        this.detectionPipeline.setDetectionCallback((detections, fps) => {
            this.updateDetectionStats(detections, fps);

            // 보정 시도 및 UI 업데이트
            this.calibrationController.update(detections);
            this.updateCalibrationStatusUI(this.calibrationController.getStatus());
        });
    }
    
    updateDetectionStats(detections, fps) {
        if (this.detectionFps) {
            this.detectionFps.textContent = fps;
        }
        if (this.detectedObjects) {
            this.detectedObjects.textContent = detections.length;
        }
    }
    
    updateCalibrationStatusUI(status) {
        if (!this.calibrationStatus) return;

        if (status.isCalibrated) {
            this.calibrationStatus.textContent = `보정 완료 (${status.calibrationObject})`;
            this.calibrationStatus.className = 'stat-value calibrated';
            this.calibrationValue.textContent = `${status.mmPerPixel.toFixed(4)} mm/px`;
        } else {
            this.calibrationStatus.textContent = '기준 객체 찾는 중...';
            this.calibrationStatus.className = 'stat-value not-calibrated';
            this.calibrationValue.textContent = 'N/A';
        }
    }
    
    showDetectionControls() {
        if (this.detectionControls) {
            this.detectionControls.style.display = 'block';
        }
    }
    
    hideDetectionControls() {
        if (this.detectionControls) {
            this.detectionControls.style.display = 'none';
        }
    }
    
    async startCamera() {
        try {
            this.updateStatus('카메라 접근 권한을 요청 중...', 'loading');
            this.startButton.disabled = true;
            
            // 카메라 제약 조건 설정
            const constraints = {
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'environment' // 후면 카메라 우선 (모바일)
                },
                audio: false
            };
            
            // getUserMedia API를 사용하여 카메라 스트림 요청
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            
            // 비디오 요소에 스트림 연결
            this.video.srcObject = this.stream;
            
            // 버튼 상태 업데이트
            this.startButton.disabled = true;
            this.stopButton.disabled = false;
            
            this.updateStatus('카메라 스트림을 설정 중...', 'loading');
            
            // 비디오 메타데이터 로드 대기
            await new Promise((resolve) => {
                this.video.addEventListener('loadedmetadata', resolve, { once: true });
            });
            
            this.isStreaming = true;
            this.updateStatus('카메라가 성공적으로 시작되었습니다!', 'success');
            
            // AI 모델 로딩 시작
            this.updateStatus('AI 모델을 로딩 중입니다...', 'loading');
            try {
                await this.objectDetectionModel.loadModel();
                this.updateStatus('AI 모델 로딩 완료! 객체 탐지를 시작합니다...', 'success');
                
                // 객체 탐지 파이프라인 시작
                this.detectionPipeline.startDetection();
                this.showDetectionControls();
                this.updateStatus('카메라와 AI 객체 탐지가 모두 준비되었습니다!', 'success');
                
            } catch (modelError) {
                console.error('AI 모델 로딩 실패:', modelError);
                this.updateStatus('카메라는 시작되었지만 AI 모델 로딩에 실패했습니다.', 'error');
            }
            
        } catch (error) {
            console.error('카메라 접근 오류:', error);
            this.handleCameraError(error);
        }
    }
    
    stopCamera() {
        try {
            // 스트림 중지
            if (this.stream) {
                this.stream.getTracks().forEach(track => {
                    track.stop();
                });
                this.stream = null;
            }
            
            // 비디오 요소 초기화
            this.video.srcObject = null;
            
            // 객체 탐지 파이프라인 중지
            if (this.detectionPipeline) {
                this.detectionPipeline.dispose();
            }
            
            // 탐지 컨트롤 숨기기
            this.hideDetectionControls();
            
            // AI 모델 정리
            if (this.objectDetectionModel) {
                this.objectDetectionModel.dispose();
            }
            
            // 상태 업데이트
            this.isStreaming = false;
            this.startButton.disabled = false;
            this.stopButton.disabled = true;
            
            this.updateStatus('카메라가 중지되었습니다', 'default');
            
        } catch (error) {
            console.error('카메라 중지 오류:', error);
            this.updateStatus('카메라 중지 중 오류가 발생했습니다', 'error');
        }
    }
    
    handleCameraError(error) {
        let errorMessage = '카메라 접근 중 오류가 발생했습니다';
        
        switch (error.name) {
            case 'NotAllowedError':
                errorMessage = '카메라 접근 권한이 거부되었습니다. 브라우저 설정에서 카메라 권한을 허용해주세요.';
                break;
            case 'NotFoundError':
                errorMessage = '카메라를 찾을 수 없습니다. 카메라가 연결되어 있는지 확인해주세요.';
                break;
            case 'NotReadableError':
                errorMessage = '카메라가 다른 애플리케이션에서 사용 중입니다. 다른 애플리케이션을 종료하고 다시 시도해주세요.';
                break;
            case 'OverconstrainedError':
                errorMessage = '카메라 설정을 만족할 수 없습니다. 다른 카메라를 사용해주세요.';
                break;
            case 'SecurityError':
                errorMessage = '보안상의 이유로 카메라에 접근할 수 없습니다. HTTPS 연결을 사용해주세요.';
                break;
            case 'TypeError':
                errorMessage = 'getUserMedia API를 지원하지 않는 브라우저입니다. 최신 브라우저를 사용해주세요.';
                break;
            default:
                errorMessage = `카메라 오류: ${error.message}`;
        }
        
        this.updateStatus(errorMessage, 'error');
        this.startButton.disabled = false;
        this.stopButton.disabled = true;
    }
    
    updateStatus(message, type = 'default') {
        this.statusMessage.textContent = message;
        this.statusMessage.className = `status-message ${type}`;
        
        // 로딩 상태일 때 애니메이션 추가
        if (type === 'loading') {
            this.statusMessage.classList.add('loading');
        } else {
            this.statusMessage.classList.remove('loading');
        }
    }
    
    // 카메라 스트림 상태 확인
    isCameraActive() {
        return this.isStreaming && this.stream && this.stream.active;
    }
    
    // 현재 비디오 프레임을 캔버스에 그리기 (향후 객체 인식에 사용)
    captureFrame() {
        if (!this.isCameraActive()) {
            throw new Error('카메라가 활성화되지 않았습니다');
        }
        
        const context = this.canvas.getContext('2d');
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        
        context.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        return this.canvas;
    }
    
    // 비디오 스트림 정보 가져오기
    getStreamInfo() {
        if (!this.stream) {
            return null;
        }
        
        const videoTrack = this.stream.getVideoTracks()[0];
        if (!videoTrack) {
            return null;
        }
        
        const settings = videoTrack.getSettings();
        return {
            width: settings.width,
            height: settings.height,
            frameRate: settings.frameRate,
            deviceId: settings.deviceId,
            label: videoTrack.label
        };
    }
}

// 브라우저 호환성 확인
function checkBrowserCompatibility() {
    const isHTTPS = location.protocol === 'https:' || location.hostname === 'localhost';
    const hasGetUserMedia = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    
    if (!isHTTPS) {
        console.warn('HTTPS 연결이 필요합니다. 일부 브라우저에서는 카메라 접근이 제한될 수 있습니다.');
    }
    
    if (!hasGetUserMedia) {
        console.error('getUserMedia API를 지원하지 않는 브라우저입니다.');
        document.getElementById('statusMessage').textContent = 
            '이 브라우저는 카메라 접근을 지원하지 않습니다. 최신 브라우저를 사용해주세요.';
        document.getElementById('startButton').disabled = true;
        return false;
    }
    
    return true;
}

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    console.log('AI 실시간 객체 칼로리 추정 서비스 초기화 중...');
    
    // 브라우저 호환성 확인
    if (!checkBrowserCompatibility()) {
        return;
    }
    
    // 카메라 컨트롤러 초기화
    window.cameraController = new CameraController();
    
    console.log('초기화 완료');
});

// 페이지 언로드 시 카메라 정리
window.addEventListener('beforeunload', () => {
    if (window.cameraController) {
        window.cameraController.stopCamera();
    }
});

// 에러 핸들링
window.addEventListener('error', (event) => {
    console.error('전역 에러:', event.error);
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('처리되지 않은 Promise 거부:', event.reason);
});

