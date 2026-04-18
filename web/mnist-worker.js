export function useMNISTClassifier() {
    let session = null
    let isProcessing = false
    let backendInfo = 'Initializing...'
    
    const loadModel = async (modelPath) => {
        try {
            const hasWebGPU = typeof navigator !== 'undefined' && 'gpu' in navigator
            
            let providers = ['wasm']
            if (hasWebGPU) {
                providers = ['webgpu', 'wasm']
            }
            
            const options = {
                executionProviders: providers,
                graphOptimizationLevel: 'all'
            }
            
            session = await ort.InferenceSession.create(modelPath, options)
            
            const provider = session.handler?.session?.getExecutionProviders?.() || 
                           session.providers || 
                           (hasWebGPU ? ['webgpu'] : ['wasm'])
            
            backendInfo = provider[0] === 'webgpu' ? 'WebGPU (GPU)' : 'WebAssembly (CPU)'
            
            console.log(`✅ Model loaded with ${backendInfo}`)
            return true
            
        } catch (error) {
            console.error('Failed to load model:', error)
            return false
        }
    }
    
    const preprocessCanvas = (canvasElement) => {
        const tempCanvas = document.createElement('canvas')
        tempCanvas.width = 28
        tempCanvas.height = 28
        const ctx = tempCanvas.getContext('2d')
        
        ctx.fillStyle = 'black'
        ctx.fillRect(0, 0, 28, 28)
        ctx.drawImage(canvasElement, 0, 0, 28, 28)
        
        const imageData = ctx.getImageData(0, 0, 28, 28)
        const data = imageData.data
        
        const pixels = new Float32Array(1 * 1 * 28 * 28)
        
        for (let i = 0; i < 28 * 28; i++) {
            const gray = (data[i * 4] + data[i * 4 + 1] + data[i * 4 + 2]) / 3
            let normalized = gray / 255.0
            normalized = (normalized - 0.1307) / 0.3081
            pixels[i] = normalized
        }
        
        return new ort.Tensor('float32', pixels, [1, 1, 28, 28])
    }
    
    const preprocessImage = async (imageFile) => {
        return new Promise((resolve, reject) => {
            const img = new Image()
            img.onload = () => {
                const tempCanvas = document.createElement('canvas')
                tempCanvas.width = 28
                tempCanvas.height = 28
                const ctx = tempCanvas.getContext('2d')
                
                ctx.fillStyle = 'black'
                ctx.fillRect(0, 0, 28, 28)
                ctx.drawImage(img, 0, 0, 28, 28)
                
                const imageData = ctx.getImageData(0, 0, 28, 28)
                const data = imageData.data
                
                let sum = 0
                for (let i = 0; i < 28 * 28; i++) {
                    const gray = (data[i * 4] + data[i * 4 + 1] + data[i * 4 + 2]) / 3
                    sum += gray / 255.0
                }
                const mean = sum / (28 * 28)
                
                const pixels = new Float32Array(1 * 1 * 28 * 28)
                
                for (let i = 0; i < 28 * 28; i++) {
                    let gray = (data[i * 4] + data[i * 4 + 1] + data[i * 4 + 2]) / 3
                    let normalized = gray / 255.0
                    
                    if (mean > 0.5) {
                        normalized = 1.0 - normalized
                    }
                    
                    normalized = (normalized - 0.1307) / 0.3081
                    pixels[i] = normalized
                }
                
                resolve(new ort.Tensor('float32', pixels, [1, 1, 28, 28]))
            }
            img.onerror = reject
            img.src = URL.createObjectURL(imageFile)
        })
    }
    
    const predict = async (input) => {
        if (!session) {
            throw new Error('Model not loaded')
        }
        
        if (isProcessing) {
            return null
        }
        
        isProcessing = true
        
        try {
            let inputTensor
            if (input instanceof HTMLCanvasElement) {
                inputTensor = preprocessCanvas(input)
            } else if (input instanceof File) {
                inputTensor = await preprocessImage(input)
            } else {
                throw new Error('Unsupported input type')
            }
            
            const feeds = { input: inputTensor }
            const results = await session.run(feeds)
            const output = results.output
            
            let outputData = output.data
            
            const expValues = Array.from(outputData).map(x => Math.exp(x))
            const sumExp = expValues.reduce((a, b) => a + b, 0)
            const probabilities = expValues.map(x => x / sumExp)
            
            const predictedDigit = probabilities.indexOf(Math.max(...probabilities))
            const confidence = probabilities[predictedDigit]
            
            return {
                digit: predictedDigit,
                confidence: confidence,
                probabilities: probabilities,
                backend: backendInfo
            }
            
        } finally {
            isProcessing = false
        }
    }
    
    const release = () => {
        if (session) {
            session.release()
            session = null
        }
    }
    
    return {
        loadModel,
        predict,
        release,
        get backend() { return backendInfo },
        get isProcessing() { return isProcessing }
    }
}