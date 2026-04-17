from dataclasses import dataclass
import ipaddress

@dataclass
class Booth:
    booth_name: str = ""
    model_4100_ip_address: str = ""
    model_4100_pin: int = 1001

    def is_valid_ip (self) -> bool:
        try:
            ipaddress.ip_address(self.model_4100_ip_address)
            return True
        except ValueError:
            return False
    
    def is_valid_pin (self) -> bool:
        pin_str: str = str(self.model_4100_pin)
        return (len(pin_str) == 4)
    