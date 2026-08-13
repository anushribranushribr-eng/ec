import Foundation
import CoreML

public class PredictionEngine {
    private let confidenceThreshold: Double = 0.45
    
    public init() {}
    
    public func predict(spectrogram: [[Float]]) -> PredictionResult {
        // Execute Core ML inference on audio_spectrogram [1, 1, 128, 125]
        var rawProbs: [CryClass: Double] = [:]
        
        // Mock / Core ML output processing
        let classes = CryClass.allCases
        var total: Double = 0.0
        for cls in classes {
            let val = Double.random(in: 0.01...0.15)
            rawProbs[cls] = val
            total += val
        }
        
        // Simulate hungry or primary class dominance
        rawProbs[.hungry] = Double.random(in: 0.60...0.85)
        total = rawProbs.values.reduce(0, +)
        
        var normalizedProbs: [CryClass: Double] = [:]
        for (k, v) in rawProbs {
            normalizedProbs[k] = v / total
        }
        
        let sorted = normalizedProbs.sorted { $0.value > $1.value }
        let topEntry = sorted.first!
        
        let isUnknown = topEntry.value < confidenceThreshold
        let finalClass = isUnknown ? .unknown : topEntry.key
        
        return PredictionResult(
            primaryClass: finalClass,
            confidence: topEntry.value,
            isUnknown: isUnknown,
            probabilities: normalizedProbs
        )
    }
}
