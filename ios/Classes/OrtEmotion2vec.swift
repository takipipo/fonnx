import Flutter
import onnxruntime_objc
import os

class OrtEmotion2Vec {
  var modelPath: String
  lazy var sessionObjects: OrtSessionObjects = {
    OrtSessionObjects(modelPath: modelPath, includeOrtExtensions: false)!
  }()


  init(modelPath: String) {
    self.modelPath = modelPath
  }

  func getEmotion(audioData: [Float]) -> [Float]? {
    os_log("Processing audio data for emotion detection, size: %d", audioData.count)
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
      
      // Run Emotion2Vec model
      let emotion2vecOutputs = try model.session.run(
        withInputs: ["input": inputTensor],
        outputNames: Set(["output"]),
        runOptions: nil
      )
      
      guard let emotion2vecOutput = emotion2vecOutputs["output"] else {
        os_log("Failed to get emotion2vec output")
        return nil
      }
      
      // Create padding mask (all true since we're using the full sequence)
      let paddingMask = Array(repeating: true, count: audioData.count)
      let paddingMaskNS = NSMutableData(
        bytes: paddingMask,
        length: paddingMask.count * MemoryLayout<Bool>.size
      )
      
      let paddingMaskTensor = try ORTValue(
        tensorData: paddingMaskNS,
        elementType: .bool,
        shape: [NSNumber(value: 1), NSNumber(value: paddingMask.count)]
      )
      
      // Run Classifier model
      let classifierOutputs = try model.session.run(
        withInputs: [
          "input": emotion2vecOutput,
          "padding_mask": paddingMaskTensor
        ],
        outputNames: Set(["output"]),
        runOptions: nil
      )
      
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