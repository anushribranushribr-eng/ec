import SwiftUI

public struct ProbabilityBarsView: View {
    public let probabilities: [CryClass: Double]
    
    public var sortedEntries: [(key: CryClass, value: Double)] {
        probabilities.sorted { $0.value > $1.value }
    }
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Class Probabilities")
                .font(.headline)
                .foregroundColor(.secondary)
            
            ForEach(sortedEntries, id: \.key) { entry in
                HStack {
                    Text(entry.key.displayName)
                        .font(.subheadline)
                        .frame(width: 140, alignment: .leading)
                    
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: 4)
                                .fill(Color.gray.opacity(0.15))
                            
                            RoundedRectangle(cornerRadius: 4)
                                .fill(entry.key == sortedEntries.first?.key ? Color.blue : Color.gray.opacity(0.6))
                                .frame(width: geo.size.width * CGFloat(entry.value))
                        }
                    }
                    .frame(height: 12)
                    
                    Text("\(Int(entry.value * 100))%")
                        .font(.caption)
                        .bold()
                        .frame(width: 40, alignment: .trailing)
                }
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 12).fill(Color(.secondarySystemBackground)))
    }
}
