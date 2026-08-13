import Foundation

public enum CryClass: String, CaseIterable, Codable, Identifiable {
    case bellyPain = "belly_pain"
    case burping = "burping"
    case coldHot = "cold_hot"
    case discomfort = "discomfort"
    case hungry = "hungry"
    case lonely = "lonely"
    case scared = "scared"
    case tired = "tired"
    case unknown = "unknown"
    
    public var id: String { rawValue }
    
    public var displayName: String {
        switch self {
        case .bellyPain: return "Belly Pain"
        case .burping: return "Burping Needed"
        case .coldHot: return "Temperature Discomfort"
        case .discomfort: return "General Discomfort"
        case .hungry: return "Hungry"
        case .lonely: return "Needs Affection / Lonely"
        case .scared: return "Startled / Scared"
        case .tired: return "Sleepy / Tired"
        case .unknown: return "Uncertain / Noise"
        }
    }
    
    public var neutralDescription: String {
        switch self {
        case .bellyPain: return "May correspond to an acoustic pattern associated with gas or belly sensitivity."
        case .burping: return "May correspond to a pattern associated with needing to burp after feeding."
        case .coldHot: return "May correspond to a temperature change or clothing discomfort."
        case .discomfort: return "May correspond to a general discomfort or wet diaper."
        case .hungry: return "May correspond to a rhythmic acoustic pattern associated with hunger."
        case .lonely: return "May correspond to a desire for physical closeness or soothe contact."
        case .scared: return "May correspond to a sudden distress vocalization."
        case .tired: return "May correspond to a lower frequency rhythmic cry associated with tiredness."
        case .unknown: return "The acoustic pattern does not match supported categories with high confidence."
        }
    }
}

public struct PredictionResult: Identifiable, Codable {
    public let id: UUID
    public let timestamp: Date
    public let primaryClass: CryClass
    public let confidence: Double
    public let isUnknown: Bool
    public let probabilities: [CryClass: Double]
    public let safetyDisclaimer: String
    
    public init(
        id: UUID = UUID(),
        timestamp: Date = Date(),
        primaryClass: CryClass,
        confidence: Double,
        isUnknown: Bool,
        probabilities: [CryClass: Double],
        safetyDisclaimer: String = "AI prediction only — always check your baby's actual needs."
    ) {
        self.id = id
        self.timestamp = timestamp
        self.primaryClass = primaryClass
        self.confidence = confidence
        self.isUnknown = isUnknown
        self.probabilities = probabilities
        self.safetyDisclaimer = safetyDisclaimer
    }
}
