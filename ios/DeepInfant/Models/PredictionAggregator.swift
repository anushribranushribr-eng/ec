import Foundation

public class PredictionAggregator {
    private let alpha: Double = 0.35
    private var smoothedProbs: [CryClass: Double] = [:]
    
    public init() {
        reset()
    }
    
    public func reset() {
        let count = Double(CryClass.allCases.count)
        for cls in CryClass.allCases {
            smoothedProbs[cls] = 1.0 / count
        }
    }
    
    public func update(with result: PredictionResult) -> PredictionResult {
        var newSmoothed: [CryClass: Double] = [:]
        var total: Double = 0.0
        
        for cls in CryClass.allCases {
            let current = result.probabilities[cls] ?? 0.0
            let prev = smoothedProbs[cls] ?? (1.0 / 9.0)
            let smoothed = alpha * current + (1.0 - alpha) * prev
            newSmoothed[cls] = smoothed
            total += smoothed
        }
        
        // Normalize
        for (k, v) in newSmoothed {
            newSmoothed[k] = v / total
        }
        smoothedProbs = newSmoothed
        
        let sorted = newSmoothed.sorted { $0.value > $1.value }
        let topEntry = sorted.first!
        let isUnknown = topEntry.value < 0.45
        
        return PredictionResult(
            primaryClass: isUnknown ? .unknown : topEntry.key,
            confidence: topEntry.value,
            isUnknown: isUnknown,
            probabilities: newSmoothed
        )
    }
}
