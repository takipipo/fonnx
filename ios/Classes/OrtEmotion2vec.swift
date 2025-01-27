import Flutter
import onnxruntime_objc
import os

class OrtEmotion2Vec {
  let logger = OSLog(subsystem: "com.fonnx.emotion2vec", category: "emotion2vec")
  
  var emotion2vecModelPath: String
  var classifierModelPath: String
  
  lazy var emotion2vecSessionObjects: OrtSessionObjects = {
    os_log(.debug, log: logger, "Initializing emotion2vec model at path: %{public}s", emotion2vecModelPath)
    guard FileManager.default.fileExists(atPath: emotion2vecModelPath) else {
      os_log(.error, log: logger, "Emotion2vec model file not found at path: %{public}s", emotion2vecModelPath)
      fatalError("Emotion2vec model file not found at path: \(emotion2vecModelPath)")
    }
    let session = OrtSessionObjects(modelPath: emotion2vecModelPath, includeOrtExtensions: false)!
    os_log(.debug, log: logger, "Successfully initialized emotion2vec session")
    return session
  }()
  
  lazy var classifierSessionObjects: OrtSessionObjects = {
    os_log(.debug, log: logger, "Initializing classifier model at path: %{public}s", classifierModelPath)
    guard FileManager.default.fileExists(atPath: classifierModelPath) else {
      os_log(.error, log: logger, "Classifier model file not found at path: %{public}s", classifierModelPath)
      fatalError("Classifier model file not found at path: \(classifierModelPath)")
    }
    let session = OrtSessionObjects(modelPath: classifierModelPath, includeOrtExtensions: false)!
    os_log(.debug, log: logger, "Successfully initialized classifier session")
    return session
  }()

  init(emotion2vecModelPath: String, classifierModelPath: String) {
    self.emotion2vecModelPath = emotion2vecModelPath
    self.classifierModelPath = classifierModelPath
  }

  func getEmotion(audioData: [Float]) -> [Float]? {
    os_log("Processing audio data for emotion detection, size: %d", audioData.count)
    
    let emotion2vecSession = emotion2vecSessionObjects.session
    let classifierSession = classifierSessionObjects.session
    
    do {
      // Create input tensor from audio data
      let audioDataNS = NSMutableData(
        bytes: audioData,
        length: audioData.count * MemoryLayout<Float>.size
      )
      
      let inputTensor = try ORTValue(
        tensorData: audioDataNS,
        elementType: .float,
        shape: [NSNumber(value: 1), NSNumber(value: audioData.count)]
      )
        // Print input tensor shape
        let tensorInfo = try inputTensor.tensorTypeAndShapeInfo()
        let inputShape = tensorInfo.shape
        os_log("Audio input tensor shape: %{public}@", inputShape.description)
      
      let startTime = CACurrentMediaTime()
      // Run Emotion2Vec model
      let emotion2vecOutputs = try emotion2vecSession.run(
        withInputs: ["input": inputTensor],
        outputNames: Set(["output"]),
        runOptions: nil
      )
      
      guard let emotion2vecOutput = emotion2vecOutputs["output"] else {
        os_log("Failed to get emotion2vec output")
        return nil
      }
      // Print emotion2vec output shape
      let outputShapeInfo = try emotion2vecOutput.tensorTypeAndShapeInfo()
      let outputShape = outputShapeInfo.shape
      let paddingMaskLength = outputShape[1].intValue
      os_log("Emotion2vec output tensor shape: %{public}@", outputShape.description)
      
      // Create padding mask (all true since we're using the full sequence)
      let paddingMask = Array(repeating: UInt8(1), count: paddingMaskLength)
      let paddingMaskNS = NSMutableData(
        bytes: paddingMask,
        length: paddingMask.count * MemoryLayout<UInt8>.size
      )
      
      let paddingMaskTensor = try ORTValue(
        tensorData: paddingMaskNS,
        elementType: .uInt8,
        shape: [NSNumber(value: 1), NSNumber(value: paddingMask.count)]
      ) 
      
      // Run Classifier model
      let classifierOutputs = try classifierSession.run(
        withInputs: [
          "input": emotion2vecOutput,
          "padding_mask": paddingMaskTensor
        ],
        outputNames: Set(["output"]),
        runOptions: nil
      )
      let endTime = CACurrentMediaTime()
      os_log("Classifier outputs: %{public}@", classifierOutputs)
      os_log("Inference latency: %.2f ms", endTime - startTime)
      guard let outputTensor = classifierOutputs["output"],
            let outputData = try? outputTensor.tensorData() as Data else {
        os_log("Failed to get classifier output")
        return nil
      }
      
      // Convert output data to array of floats
      return outputData.toArray(type: Float.self)
      
    } catch {
      os_log("Error in getEmotion: %{public}s", error.localizedDescription)
      for symbol in Thread.callStackSymbols {
        os_log("Stack trace: %{public}s", symbol)
      }
      return nil
    }
  }
}