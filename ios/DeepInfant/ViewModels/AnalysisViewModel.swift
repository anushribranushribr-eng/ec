import Foundation
import Combine

public class AnalysisViewModel: ObservableObject {
    @Published public var isListening: Bool = false
    @Published public var audioLevel: Float = 0.0
    @Published public var currentResult: PredictionResult?
    @Published public var history: [PredictionResult] = []
    
    public let audioManager = AudioManager()
    private let engine = PredictionEngine()
    private let aggregator = PredictionAggregator()
    
    public init() {
        audioManager.onAudioChunkProcessed = { [weak self] spectrogram in
            guard let self = self else { return }
            let rawResult = self.engine.predict(spectrogram: spectrogram)
            let smoothedResult = self.aggregator.update(with: rawResult)
            
            DispatchQueue.main.async {
                self.currentResult = smoothedResult
                self.audioLevel = self.audioManager.currentAudioLevel
            }
        }
    }
    
    public func toggleListening() {
        if isListening {
            audioManager.stopListening()
            isListening = false
            if let result = currentResult {
                history.insert(result, at: 0)
            }
        } else {
            aggregator.reset()
            audioManager.startListening()
            isListening = true
        }
    }
}
