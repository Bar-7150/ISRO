"""
api_server.py
Lightweight Edge-AI REST API Service for ISRO Burn-In Screening.
Connects the Next.js Web Dashboard directly to:
1. Python Module A (AEC-Q001 DPATEngine & GDBNSpatialEngine)
2. Python Module B (PhysicsInformedDriftPredictor - XGBoost & Bayesian Ridge)
3. Python Hardware Edge Controller (INA219 I2C current sensor & GPIO 17 Relay)
4. TreeSHAP & Surrogate Rule Explainer

Zero external server dependencies required (uses Python built-in http.server).
"""

import os
import sys
import json
import time
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))

# UTF-8 console output for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] API: %(message)s")
logger = logging.getLogger("ISRO_Edge_API")

from src.module_a_outlier_detection import DPATEngine, GDBNSpatialEngine
from src.module_b_drift_predictor import (
    PhysicsInformedDriftPredictor,
    calculate_arrhenius_acceleration,
    calculate_electromigration_af,
    calculate_safety_slope,
    calculate_arrhenius_thermo_floor,
    compute_pinn_arrhenius_loss
)
from src.hardware_edge_controller import HardwareEdgeController
from src.as9100_compliance import AS9100CertificateGenerator
from src.federated_edge_mesh import FederatedEdgeNode, FederatedMeshCoordinator

# Initialize persistent Python Edge engines
drift_predictor = PhysicsInformedDriftPredictor(safety_limit_168h=25.0, alpha_conformal=0.001)
hardware_controller = HardwareEdgeController()
cert_generator = AS9100CertificateGenerator()
mesh_coordinator = FederatedMeshCoordinator()

# Pre-seed 3 active factory edge chambers
node1 = FederatedEdgeNode("CHAMBER-01", "Bengaluru Cleanroom Fab-1")
node2 = FederatedEdgeNode("CHAMBER-02", "Sriharikota QA Screening Bay")
node3 = FederatedEdgeNode("CHAMBER-03", "Thiruvananthapuram VSSC Edge Cell")
node1.record_burn_in_session(180, 24.0, 0.0132, 0.705)
node2.record_burn_in_session(250, 24.0, 0.0126, 0.698)
node3.record_burn_in_session(140, 24.0, 0.0140, 0.710)
mesh_coordinator.register_node(node1)
mesh_coordinator.register_node(node2)
mesh_coordinator.register_node(node3)


class EdgeAPIRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests from the Next.js frontend."""

    def _set_headers(self, status_code: int = 200, content_type: str = "application/json"):
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(204)

    def do_GET(self):
        if self.path == "/api/status":
            telemetry = hardware_controller.read_sensors()
            payload = {
                "service": "ISRO Edge-AI Python ML Server",
                "status": "ONLINE",
                "connected_models": [
                    "AEC-Q001 DPATEngine (Spatial Clustering & Z-Score)",
                    "PhysicsInformedDriftPredictor (XGBoost + Arrhenius PINN Loss)",
                    "ConformalDriftPredictor (99.9% Confidence, FN <= 0.01%)",
                    "HardwareEdgeController (INA219 Auto-Zero & Shunt Thermal Compensation)",
                    "AS9100 Digital Certificate Generator (JSON & PDF)",
                    "Federated Edge Learning Mesh (FedAvg across 3 Cleanroom Chambers)"
                ],
                "telemetry": telemetry,
                "mesh_summary": {
                    "active_chambers": len(mesh_coordinator.nodes),
                    "total_network_hours": mesh_coordinator.total_network_test_hours,
                    "round": mesh_coordinator.current_round
                },
                "timestamp": time.time()
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(payload).encode("utf-8"))

        elif self.path == "/api/telemetry":
            hardware_controller.step_simulation_time(0.1)
            telemetry = hardware_controller.read_sensors()
            self._set_headers(200)
            self.wfile.write(json.dumps(telemetry).encode("utf-8"))

        elif self.path == "/api/federated/mesh_status":
            status = mesh_coordinator.get_mesh_status()
            self._set_headers(200)
            self.wfile.write(json.dumps(status).encode("utf-8"))

        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_length)
        data = json.loads(post_body.decode("utf-8")) if post_body else {}

        if self.path in ("/api/predict_drift", "/api/predict_conformal"):
            # Conformal Prediction + PINN Thermodynamics Forecast
            val_0h = float(data.get("val_0h", 10.0))
            val_24h = float(data.get("val_24h", 18.5))
            temp_c = float(data.get("temp_c", 125.0))
            voltage_v = float(data.get("voltage_v", 3.6))
            safety_limit = float(data.get("safety_limit_168h", 25.0))

            drift_predictor.safety_limit_168h = safety_limit
            result = drift_predictor.forecast_trajectory(
                val_0h=val_0h,
                val_24h=val_24h,
                temp_c=temp_c,
                voltage_v=voltage_v
            )
            result["engine"] = "Python XGBoost Quantile Regressor + Conformal Prediction (99.9% Confidence)"
            self._set_headers(200)
            self.wfile.write(json.dumps(result).encode("utf-8"))

        elif self.path == "/api/sensor_calibrate":
            # Auto-Zero & Shunt Thermal Compensation
            ambient_temp = float(data.get("ambient_temp_c", 25.0))
            cal_res = hardware_controller.auto_zero_calibrate(ambient_temp_c=ambient_temp)
            current_tele = hardware_controller.read_sensors()
            payload = {
                "calibration_result": cal_res,
                "current_telemetry": current_tele
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(payload).encode("utf-8"))

        elif self.path == "/api/compliance/generate_certificate":
            # AS9100 / MIL-STD-883 Digital Birth Certificate
            chip_id = str(data.get("chip_serial_id", f"ISRO-CHIP-{int(time.time()) % 100000:05d}"))
            socket_id = str(data.get("socket_id", "CHAMBER-01/TRAY-01/SOCKET-C4"))
            lot_id = str(data.get("lot_id", "ISRO-LOT-2026-A1"))
            lot_mean = float(data.get("lot_mean", 10.0))
            lot_std = float(data.get("lot_std", 2.5))
            val_0h = float(data.get("val_0h", 9.85))
            val_24h = float(data.get("val_24h", 10.35))
            pred_168h = float(data.get("pred_168h", 11.20))
            conf_upper = float(data.get("conformal_upper_168h", 12.85))

            cert = cert_generator.create_certificate(
                chip_serial_id=chip_id,
                socket_id=socket_id,
                lot_id=lot_id,
                lot_mean=lot_mean,
                lot_std=lot_std,
                val_0h=val_0h,
                val_24h=val_24h,
                pred_168h_point=pred_168h,
                conformal_upper_168h=conf_upper
            )
            json_file = cert_generator.export_json(cert)
            pdf_file = cert_generator.export_pdf(cert)

            response_data = {
                "success": True,
                "certificate_id": cert["certificate_header"]["certificate_id"],
                "sha256_hash": cert["cryptographic_verification"]["tamper_evident_hash"],
                "json_path": json_file,
                "pdf_path": pdf_file,
                "certificate": cert
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(response_data).encode("utf-8"))

        elif self.path == "/api/federated/trigger_round":
            # Execute FedAvg cycle across edge chambers
            round_res = mesh_coordinator.execute_federated_round()
            self._set_headers(200)
            self.wfile.write(json.dumps(round_res).encode("utf-8"))

        elif self.path == "/api/dpat_screen":
            # Module A: Real Python DPAT screening
            lot_mean = float(data.get("lot_mean", 10.0))
            lot_std = float(data.get("lot_std", 2.5))
            candidate_val = float(data.get("candidate_val", 45.0))
            static_limit = float(data.get("static_limit", 50.0))
            k_sigma = float(data.get("k_sigma", 3.0))

            bench = DPATEngine.evaluate_latent_defect_benchmark(
                lot_mean=lot_mean,
                lot_std=lot_std,
                candidate_value=candidate_val,
                static_spec_limit=static_limit,
                k_sigma=k_sigma
            )
            bench["engine"] = "Python AEC-Q001 DPATEngine (Real ML Inference)"
            self._set_headers(200)
            self.wfile.write(json.dumps(bench).encode("utf-8"))

        elif self.path == "/api/relay_trip":
            reason = str(data.get("reason", "API Eject Command"))
            event = hardware_controller.trip_relay(reason=reason)
            self._set_headers(200)
            self.wfile.write(json.dumps({"success": True, "event": event}).encode("utf-8"))

        elif self.path == "/api/relay_reset":
            event = hardware_controller.reset_relay()
            self._set_headers(200)
            self.wfile.write(json.dumps({"success": True, "event": event}).encode("utf-8"))

        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

    def log_message(self, format, *args):
        # Concise logging
        logger.info(f"{self.address_string()} - {format % args}")


def start_server(host: str = "0.0.0.0", port: int = 5000):
    server = HTTPServer((host, port), EdgeAPIRequestHandler)
    logger.info(f"🛰️ ISRO Edge-AI Python ML Server listening on http://{host}:{port}")
    logger.info(f"   Connected: Module A (DPAT), Module B (Conformal + PINN), INA219 Auto-Zero, AS9100, FedMesh")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped.")
        server.server_close()


if __name__ == "__main__":
    start_server(port=5000)

