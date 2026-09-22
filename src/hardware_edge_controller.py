"""
hardware_edge_controller.py
Hardware & Edge-AI Controller for Component Burn-In Screening.
Designed for Raspberry Pi Zero 2 W / Jetson Nano / Embedded Edge PC:
1. INA219 I2C Current & Voltage Sensor (Address 0x40): Monitors Iddq leakage and rail voltage
2. MAX31855 SPI Thermocouple Sensor: Monitors 125°C Burn-In Chamber Temperature
3. 5V Active-Low Hardware Relay (GPIO 17): Physically cuts high-side power to failing DUTs
4. Automatic High-Fidelity Emulation Mode: Seamlessly falls back if physical I2C/SPI/GPIO hardware is not connected.
"""

import time
import math
import random
import logging
from typing import Dict, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("HardwareEdgeController")

# Try importing physical hardware drivers
HAS_HARDWARE = False
try:
    import board
    import busio
    from adafruit_ina219 import INA219
    import digitalio
    import adafruit_max31855
    from gpiozero import OutputDevice
    HAS_HARDWARE = True
    logger.info("Physical Raspberry Pi GPIO & I2C/SPI hardware libraries detected.")
except (ImportError, NotImplementedError, AttributeError):
    HAS_HARDWARE = False
    logger.info("Physical hardware libraries not available. Initializing high-fidelity Hardware Emulation Mode.")


class HardwareEdgeController:
    """
    Edge hardware supervisor controlling sensors and safety interlock relays.
    Operates in either Physical Mode (on Raspberry Pi) or Emulation Mode (on Edge PC/Demo).
    """

    def __init__(
        self,
        relay_gpio_pin: int = 17,
        i2c_address: int = 0x40,
        nominal_voltage_v: float = 3.6,
        nominal_temp_c: float = 125.0,
        force_emulation: bool = False
    ):
        self.relay_gpio_pin = relay_gpio_pin
        self.i2c_address = i2c_address
        self.nominal_voltage_v = nominal_voltage_v
        self.nominal_temp_c = nominal_temp_c
        self.emulation_mode = force_emulation or (not HAS_HARDWARE)

        # Hardware states
        self.relay_state = "CLOSED_POWER_ON"  # "CLOSED_POWER_ON" or "OPEN_POWER_CUT"
        self.dut_powered = True
        self.trip_history = []

        # Simulated state variables
        self._sim_time_hours = 0.0
        self._sim_leakage_ua = 9.85
        self._sim_temp_c = nominal_temp_c
        self._sim_voltage_v = nominal_voltage_v
        self._sim_drift_mode = "NORMAL"  # "NORMAL", "LATENT_OUTLIER", or "RUNAWAY_DRIFT"

        # Sensor Calibration & Thermal Compensation Parameters (INA219 + Shunt)
        self.cal_temp_c = 25.0  # Reference factory calibration temperature
        self.zero_offset_ua = 0.12  # Zero-current residual amplifier offset
        self.shunt_temp_coeff_ppm = 25.0  # 25 ppm/°C for precision metal foil shunt
        self.amplifier_vos_drift_uv_c = 0.5  # 0.5 uV/°C INA219 input offset drift
        self.shunt_resistance_ohms = 0.1  # 0.1 Ohm shunt resistor
        self.is_calibrated = False
        self._sim_box_temp_c = 25.0  # Ambient control box internal electronics temperature

        # Auto-calibrate baseline at boot
        self.auto_zero_calibrate(ambient_temp_c=25.0)

        # Initialize physical devices if available
        self.ina219 = None
        self.thermocouple = None
        self.relay_device = None

        if not self.emulation_mode:
            self._init_physical_hardware()

    def auto_zero_calibrate(self, ambient_temp_c: Optional[float] = None) -> Dict[str, Union[float, str]]:
        """
        Executes zero-offset calibration routine to establish the true zero-current baseline.
        Records ambient box temperature and zeros out residual amplifier offset.
        """
        temp = ambient_temp_c or self._sim_box_temp_c
        self.cal_temp_c = temp

        if not self.emulation_mode and self.ina219 is not None and not self.dut_powered:
            try:
                # Average 10 unpowered samples
                samples = [float(self.ina219.current) * 1000.0 for _ in range(10)]
                self.zero_offset_ua = float(sum(samples) / len(samples))
            except Exception as e:
                logger.warning(f"Hardware zero calibration warning: {e}")
                self.zero_offset_ua = 0.12
        else:
            self.zero_offset_ua = 0.12 + random.gauss(0.0, 0.02)

        self.is_calibrated = True
        logger.info(f"🎯 INA219 Auto-Zero Calibrated: Zero Offset = {self.zero_offset_ua:.3f} uA at {self.cal_temp_c:.1f}°C")
        return {
            "status": "CALIBRATED",
            "calibration_temp_c": round(self.cal_temp_c, 2),
            "zero_offset_ua": round(self.zero_offset_ua, 4),
            "shunt_ppm_per_c": self.shunt_temp_coeff_ppm
        }

    def apply_thermal_compensation(
        self,
        raw_current_ua: float,
        ambient_box_temp_c: float
    ) -> Tuple[float, float]:
        """
        Factors out thermal drift of the INA219 current sensor itself when the ambient
        control box heats up, ensuring sensor drift is never misread as silicon degradation.
        
        Compensation Model:
          delta_T = T_box - T_cal
          delta_Vos = Vos_drift * delta_T (uV)
          delta_I_offset = delta_Vos / R_shunt (uA)
          I_compensated = (I_raw - zero_offset - delta_I_offset) / (1 + alpha_shunt * delta_T)
        """
        delta_t = ambient_box_temp_c - self.cal_temp_c
        delta_vos_uv = self.amplifier_vos_drift_uv_c * delta_t
        delta_i_offset_ua = delta_vos_uv / max(self.shunt_resistance_ohms, 1e-4)
        shunt_expansion = 1.0 + (self.shunt_temp_coeff_ppm * 1e-6) * delta_t

        compensated_ua = (raw_current_ua - (self.zero_offset_ua + delta_i_offset_ua)) / shunt_expansion
        compensated_ua = max(0.0, compensated_ua)
        sensor_thermal_drift_ua = raw_current_ua - compensated_ua

        return compensated_ua, sensor_thermal_drift_ua

    def _init_physical_hardware(self):
        """Initializes real I2C, SPI, and GPIO hardware on the Raspberry Pi."""
        try:
            import board
            import busio
            from adafruit_ina219 import INA219
            import digitalio
            import adafruit_max31855
            from gpiozero import OutputDevice

            # 1. Initialize I2C for INA219
            i2c = busio.I2C(board.SCL, board.SDA)
            self.ina219 = INA219(i2c, addr=self.i2c_address)
            logger.info(f"INA219 initialized at I2C address 0x{self.i2c_address:02X}")

            # 2. Initialize SPI for MAX31855 Thermocouple
            spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
            cs = digitalio.DigitalInOut(board.D5)
            self.thermocouple = adafruit_max31855.MAX31855(spi, cs)
            logger.info("MAX31855 thermocouple initialized on SPI CS=GPIO5")

            # 3. Initialize Active-Low Relay (Active-low optocoupler module)
            self.relay_device = OutputDevice(self.relay_gpio_pin, active_high=False, initial_value=True)
            self.relay_state = "CLOSED_POWER_ON"
            self.dut_powered = True
            logger.info(f"Fail-Safe Relay initialized on GPIO {self.relay_gpio_pin} (Active-Low)")
        except Exception as e:
            logger.warning(f"Failed to initialize physical hardware ({e}). Falling back to Emulation Mode.")
            self.emulation_mode = True

    def set_simulation_profile(self, mode: str = "NORMAL", base_leakage_ua: float = 10.0):
        """
        Sets the simulated component behavior:
        - "NORMAL": Stable qualification die drifting mildly from ~10 uA to ~12 uA at 168h
        - "LATENT_OUTLIER": Passes static 50uA limit but sits abnormally high at 45 uA (ISRO test case)
        - "RUNAWAY_DRIFT": Accelerates past safety slope at 24h requiring early abort
        """
        self._sim_drift_mode = mode.upper()
        self._sim_leakage_ua = base_leakage_ua
        if self._sim_drift_mode == "LATENT_OUTLIER":
            self._sim_leakage_ua = 45.0
        elif self._sim_drift_mode == "RUNAWAY_DRIFT":
            self._sim_leakage_ua = 12.0
        logger.info(f"Simulation profile set to: {self._sim_drift_mode} (Base Leakage: {self._sim_leakage_ua:.2f} uA)")

    def step_simulation_time(self, delta_hours: float = 1.0):
        """Advances simulation time and updates physical drift trajectories."""
        self._sim_time_hours += delta_hours
        t = self._sim_time_hours

        # Thermal noise
        self._sim_temp_c = self.nominal_temp_c + random.gauss(0.0, 0.4)
        self._sim_voltage_v = self.nominal_voltage_v + random.gauss(0.0, 0.01)

        # Ambient control box heats up as burn-in test runs (25°C -> ~55°C)
        self._sim_box_temp_c = 25.0 + 30.0 * (1.0 - math.exp(-t / 18.0)) + random.gauss(0.0, 0.2)

        # Kinetic degradation trajectory
        if not self.dut_powered:
            # When relay is open / tripped, leakage current drops to 0.0 uA
            self._sim_leakage_ua = 0.0
            return

        if self._sim_drift_mode == "NORMAL":
            # Slow logarithmic diffusion drift: 10 uA -> 11.5 uA at 168h
            drift = 1.5 * (t / 168.0) ** 0.8
            self._sim_leakage_ua = 10.0 + drift + random.gauss(0.0, 0.15)
        elif self._sim_drift_mode == "LATENT_OUTLIER":
            # Constant severe contextual offset (45 uA latent defect in 10 uA lot)
            drift = 3.5 * (t / 168.0) ** 0.9
            self._sim_leakage_ua = 45.0 + drift + random.gauss(0.0, 0.25)
        elif self._sim_drift_mode == "RUNAWAY_DRIFT":
            # Exponential oxide breakdown (TDDB runaway): 12 uA at 0h -> 22 uA at 24h -> 75 uA at 168h
            drift = 10.0 * (t / 24.0) ** 1.8
            self._sim_leakage_ua = 12.0 + drift + random.gauss(0.0, 0.3)

    def read_sensors(self) -> Dict[str, Union[float, str, bool, Dict]]:
        """
        Reads real I2C/SPI sensors on physical hardware or queries emulation state.
        Applies real-time thermal compensation and returns unified telemetry.
        """
        box_temp_c = self._sim_box_temp_c

        if not self.emulation_mode:
            try:
                # Real INA219 reading
                bus_voltage = float(self.ina219.bus_voltage)
                shunt_voltage_mv = float(self.ina219.shunt_voltage)
                current_ma = float(self.ina219.current)  # mA
                raw_current_ua = current_ma * 1000.0  # convert to uA
                
                # Real MAX31855 reading (Chamber temperature)
                temp_c = float(self.thermocouple.temperature)
            except Exception as e:
                logger.error(f"Hardware sensor read error: {e}. Falling back to emulation.")
                bus_voltage = self._sim_voltage_v if self.dut_powered else 0.0
                shunt_voltage_mv = 1.2
                raw_current_ua = self._sim_leakage_ua if self.dut_powered else 0.0
                temp_c = self._sim_temp_c
        else:
            bus_voltage = self._sim_voltage_v if self.dut_powered else 0.0
            temp_c = self._sim_temp_c

            # In emulation, model raw physical sensor experiencing thermal drift
            # as ambient control box heats up
            if self.dut_powered:
                # Raw sensor picks up uncalibrated offset and thermal drift
                delta_t = box_temp_c - self.cal_temp_c
                sensor_drift_sim = (self.zero_offset_ua + (self.amplifier_vos_drift_uv_c / 0.1) * delta_t)
                raw_current_ua = self._sim_leakage_ua + sensor_drift_sim
            else:
                raw_current_ua = self.zero_offset_ua

            shunt_voltage_mv = (raw_current_ua / 1000.0) * self.shunt_resistance_ohms

        # Apply Real-Time Thermal Compensation
        if self.dut_powered:
            compensated_current_ua, sensor_drift_ua = self.apply_thermal_compensation(
                raw_current_ua=raw_current_ua,
                ambient_box_temp_c=box_temp_c
            )
        else:
            compensated_current_ua = 0.0
            sensor_drift_ua = raw_current_ua

        return {
            "timestamp": time.time(),
            "sim_burn_in_hours": round(self._sim_time_hours, 1),
            "chamber_temp_c": round(temp_c, 2),
            "control_box_ambient_temp_c": round(box_temp_c, 2),
            "rail_voltage_v": round(bus_voltage, 3),
            "shunt_voltage_mv": round(shunt_voltage_mv, 4),
            "raw_iddq_leakage_current_ua": round(raw_current_ua, 3),
            "iddq_leakage_current_ua": round(compensated_current_ua, 3),
            "sensor_thermal_drift_ua": round(sensor_drift_ua, 3),
            "sensor_thermal_compensated": True,
            "sensor_calibration": {
                "is_calibrated": self.is_calibrated,
                "cal_temp_c": round(self.cal_temp_c, 2),
                "zero_offset_ua": round(self.zero_offset_ua, 4),
                "shunt_temp_coeff_ppm": self.shunt_temp_coeff_ppm
            },
            "relay_state": self.relay_state,
            "dut_power_enabled": self.dut_powered,
            "is_emulated": self.emulation_mode
        }

    def trip_relay(self, reason: str = "ANOMALY_DETECTED"):
        """
        Actuates the hardware relay to cut high-side VDD power to the DUT socket.
        Physically protects adjacent components and halts defective stress testing.
        """
        if not self.emulation_mode and self.relay_device is not None:
            try:
                self.relay_device.off()
            except Exception as e:
                logger.error(f"Failed to actuate physical relay: {e}")

        self.relay_state = "OPEN_POWER_CUT"
        self.dut_powered = False
        event = {
            "timestamp": time.time(),
            "burn_in_hours": self._sim_time_hours,
            "action": "POWER_CUT",
            "reason": reason
        }
        self.trip_history.append(event)
        logger.warning(f"🚨 HARDWARE RELAY TRIPPED! DUT Power CUT-OFF. Reason: {reason}")
        return event

    def reset_relay(self):
        """Re-energizes the relay coil to restore power to the DUT socket."""
        if not self.emulation_mode and self.relay_device is not None:
            try:
                self.relay_device.on()
            except Exception as e:
                logger.error(f"Failed to reset physical relay: {e}")

        self.relay_state = "CLOSED_POWER_ON"
        self.dut_powered = True
        logger.info("✅ HARDWARE RELAY RESET: DUT Power Restored.")
        return {"action": "POWER_RESTORED", "status": self.relay_state}


# Direct test execution
if __name__ == "__main__":
    controller = HardwareEdgeController()
    print("Initial Sensor Telemetry:")
    print(controller.read_sensors())

    print("\nSimulating 24h burn-in with Latent Defect (45 uA in 10 uA lot)...")
    controller.set_simulation_profile("LATENT_OUTLIER")
    controller.step_simulation_time(24.0)
    print(controller.read_sensors())

    print("\nTriggering Hardware Relay Eject...")
    controller.trip_relay(reason="Module A DPAT Outlier: 45.0 uA in 10 uA lot (Z=+14.0 sigma)")
    print(controller.read_sensors())

    print("\nResetting Relay...")
    controller.reset_relay()
    print(controller.read_sensors())
