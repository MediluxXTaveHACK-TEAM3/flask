from flask import Flask, request, jsonify
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
import random

app = Flask(__name__)

# 거리 계산 함수
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # 지구 반경(km)
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# 경로 계산 함수
def route(doctor_location, patients_data, emergency_calls):
    nearby_patients = []
    for patient in patients_data:
        distance = calculate_distance(
            doctor_location[0], doctor_location[1],
            patient['location'][0], patient['location'][1]
        )
        if distance <= 5:  # 5km 반경 내 환자 필터링
            nearby_patients.append(patient)

    emergency_patients = [p for p in nearby_patients if p['patientid'] in emergency_calls]

    # 방문 환자 수를 랜덤하게 결정
    current_date = datetime.now()
    random.seed(current_date.year * 10000 + current_date.month * 100 + current_date.day)
    total_patients_needed = random.randint(4, 7)

    # 경로 계획
    if len(emergency_patients) >= total_patients_needed:
        return emergency_patients

    if len(emergency_patients) > 0:
        remaining_slots = total_patients_needed - len(emergency_patients)
        regular_patients = [p for p in nearby_patients if p['patientid'] not in emergency_calls]
        selected_regular_patients = regular_patients[:remaining_slots]
        return emergency_patients + selected_regular_patients
    else:
        return nearby_patients[:total_patients_needed]

# 경로 거리 계산
def calc_route_distance(route_plan, doctor_location):
    route_distance = []
    route_distance.append(calculate_distance(
        doctor_location[0], doctor_location[1],
        route_plan[0]['location'][0], route_plan[0]['location'][1]
    ))
    for i in range(len(route_plan) - 1):
        current_location = route_plan[i]['location']
        next_location = route_plan[i + 1]['location']
        distance = calculate_distance(
            current_location[0], current_location[1],
            next_location[0], next_location[1]
        )
        route_distance.append(distance)
    return route_distance

@app.route('/plan_route', methods=['POST'])
def plan_route():
    """
    API Endpoint to plan the doctor's route.
    Expects JSON input with the doctor's location, patient data, and emergency calls.
    Returns the route plan and distances.
    """
    try:
        data = request.json
        doctor_location = data['doctor_location']
        patients = data['patients']
        emergency_calls = data['emergency_calls']

        # Calculate route and distances
        route_plan = route(doctor_location, patients, emergency_calls)
        route_distances = calc_route_distance(route_plan, doctor_location)

        # Format the response
        result = {
            "route_plan": route_plan,
            "route_distances": route_distances,
            "total_patients": len(route_plan)
        }
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(port=5001, debug=True)
