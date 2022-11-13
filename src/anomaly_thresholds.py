from dataclasses import dataclass


def find_anomalies(output_error: list[float], threshold: float):
    anomalies = []
    current_anomaly = False
    for i in range(len(output_error)):
        if output_error[i] > threshold:
            if current_anomaly:
                anomaly = anomalies[-1]
                anomaly.length += 1
                anomaly.error = max(anomaly.error, output_error[i])
            else:
                anomalies.append(
                    Anomaly(length=1, index=i, error=output_error[i]))
                current_anomaly = True
        else:
            current_anomaly = False
    return sorted(anomalies, key=lambda x: x.error, reverse=True)


def parse_anomalies(anomalies: list, timestamps: list[str]):
    return [{"timestamp": timestamps[e.index + e.length // 2], "type": "Point" if e.length == 1 else "Area"} for e in
            anomalies]
    
    
@dataclass
class Anomaly:
    length: int
    index: int
    error: float
