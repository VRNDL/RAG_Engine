# NovaTech Orbital Drone - Technical Specification & FAQ

## 1. Overview
The NovaTech OD-400 is an autonomous surveillance drone designed for extreme weather monitoring. It features a carbon-fiber unibody construction weighing 3.4 kilograms.

## 2. Battery & Range
- **Battery Pack:** 12,500 mAh Solid-State Lithium cell.
- **Flight Time:** Up to 82 minutes in normal conditions; reduced to 54 minutes in winds exceeding 40 km/h.
- **Maximum Operational Altitude:** 4,200 meters above sea level.
- **Charging Interface:** Magnetic rapid-dock connector (0% to 80% charge in 28 minutes).

## 3. Sensor Array & Compute
The drone is equipped with:
- Dual 64MP optical zoom cameras with mechanical image stabilization.
- FLIR Boson thermal camera operating at 640x512 resolution.
- On-board neural accelerator capable of 32 TOPS for real-time edge obstacle avoidance.
- Lidar unit with a 150-meter effective detection radius.

## 4. Emergency Protocols & Failsafes
- **Signal Loss (Protocol Delta):** If the command link drops for longer than 12 seconds, the drone ascends to a clear altitude of 100 meters and executes an automated return-to-home (RTH) sequence.
- **Critical Battery Alert:** At 8% remaining power, the drone initiates an immediate vertical descent regardless of location.
- **Geofence Enforcement:** The unit contains an embedded hardcoded registry of prohibited airspace zones that cannot be overridden via remote telemetry.

## 5. Maintenance Schedule
Every 150 flight hours requires rotor bearing lubrication using synthetic Krytox grease. The optical sensor glass should be cleaned exclusively with 99% isopropyl alcohol and lint-free wipes.