"""VERIS enumeration values for structured classification output."""

ACTOR_EXTERNAL_VARIETY = [
    "Activist", "Auditor", "Competitor", "Customer", "Force majeure",
    "Former employee", "Nation-state", "Organized crime", "Acquaintance",
    "State-affiliated", "Terrorist", "Unaffiliated", "Unknown", "Other",
]

ACTOR_INTERNAL_VARIETY = [
    "Auditor", "Call center", "Cashier", "End-user", "Executive", "Finance",
    "Helpdesk", "Human resources", "Maintenance", "Manager", "Guard",
    "Developer", "System admin", "Unknown", "Other",
]

ACTOR_MOTIVE = [
    "NA", "Espionage", "Fear", "Financial", "Fun", "Grudge",
    "Ideology", "Convenience", "Unknown", "Other",
]

ACTION_CATEGORIES = [
    "malware", "hacking", "social", "misuse", "physical", "error", "environmental",
]

ACTION_MALWARE_VARIETY = [
    "Adware", "Backdoor", "Brute force", "Capture app data", "Capture stored data",
    "Client-side attack", "Click fraud", "C2", "Destroy data", "Disable controls",
    "DoS", "Downloader", "Exploit vuln", "Export data", "Packet sniffer",
    "Password dumper", "Ram scraper", "Ransomware", "Rootkit", "Scan network",
    "Spam", "Spyware/Keylogger", "SQL injection", "Adminware", "Worm", "Unknown", "Other",
]

ACTION_HACKING_VARIETY = [
    "Abuse of functionality", "Brute force", "Buffer overflow", "Cache poisoning",
    "Session prediction", "CSRF", "XSS", "Cryptanalysis", "DoS", "Footprinting",
    "Forced browsing", "Format string attack", "Fuzz testing", "HTTP request smuggling",
    "HTTP request splitting", "Integer overflows", "LDAP injection", "Mail command injection",
    "MitM", "Null byte injection", "Offline cracking", "OS commanding", "Path traversal",
    "RFI", "Reverse engineering", "Routing detour", "Session fixation", "Session replay",
    "Soap array abuse", "Special element injection", "SQLi", "SSI injection",
    "URL redirector abuse", "Use of backdoor or C2", "Use of stolen creds",
    "XML attribute blowup", "XML entity expansion", "XML external entities",
    "XML injection", "XPath injection", "XQuery injection", "Virtual machine escape",
    "Unknown", "Other",
]

ACTION_SOCIAL_VARIETY = [
    "Baiting", "Bribery", "Elicitation", "Extortion", "Forgery", "Influence",
    "Scam", "Phishing", "Pretexting", "Propaganda", "Spam", "Unknown", "Other",
]

ACTION_MISUSE_VARIETY = [
    "Knowledge abuse", "Privilege abuse", "Embezzlement", "Data mishandling",
    "Email misuse", "Net misuse", "Illicit content", "Unapproved workaround",
    "Unapproved hardware", "Unapproved software", "Unknown", "Other",
]

ACTION_PHYSICAL_VARIETY = [
    "Assault", "Sabotage", "Snooping", "Surveillance", "Tampering",
    "Theft", "Wiretapping", "Unknown", "Other",
]

ACTION_ERROR_VARIETY = [
    "Classification error", "Data entry error", "Disposal error", "Gaffe", "Loss",
    "Maintenance error", "Misconfiguration", "Misdelivery", "Misinformation",
    "Omission", "Physical accidents", "Capacity shortage", "Programming error",
    "Publishing error", "Malfunction", "Unknown", "Other",
]

ASSET_VARIETY = [
    "S - Authentication", "S - Backup", "S - Database", "S - DHCP", "S - Directory",
    "S - DCS", "S - DNS", "S - File", "S - Log", "S - Mail", "S - Mainframe",
    "S - Payment switch", "S - POS controller", "S - Print", "S - Proxy",
    "S - Remote access", "S - SCADA", "S - Web application", "S - Code repository",
    "S - VM host",
    "N - Access reader", "N - Camera", "N - Firewall", "N - HSM", "N - IDS",
    "N - Broadband", "N - PBX", "N - Private WAN", "N - PLC", "N - Public WAN",
    "N - RTU", "N - Router or switch", "N - SAN", "N - Telephone", "N - VoIP adapter",
    "N - LAN", "N - WLAN",
    "U - Auth token", "U - Desktop", "U - Laptop", "U - Media", "U - Mobile phone",
    "U - Peripheral", "U - POS terminal", "U - Tablet", "U - Telephone", "U - VoIP phone",
    "T - ATM", "T - PED pad", "T - Gas terminal", "T - Kiosk",
    "M - Tapes", "M - Disk media", "M - Documents", "M - Flash drive", "M - Disk drive",
    "M - Smart card", "M - Payment card",
    "P - System admin", "P - Auditor", "P - Call center", "P - Cashier", "P - Customer",
    "P - Developer", "P - End-user", "P - Executive", "P - Finance", "P - Former employee",
    "P - Guard", "P - Helpdesk", "P - Human resources", "P - Maintenance", "P - Manager",
    "P - Partner",
    "Unknown", "Other",
]

ATTRIBUTE_CONFIDENTIALITY_DATA_VARIETY = [
    "Credentials", "Bank", "Classified", "Copyrighted", "Medical",
    "Payment", "Personal", "Internal", "System", "Secrets", "Unknown", "Other",
]

DATA_DISCLOSURE = ["Yes", "Potentially", "No", "Unknown"]

ATTRIBUTE_INTEGRITY_VARIETY = [
    "Created account", "Hardware tampering", "Alter behavior", "Fraudulent transaction",
    "Log tampering", "Misappropriation", "Misrepresentation", "Modify configuration",
    "Modify privileges", "Modify data", "Software installation", "Unknown", "Other",
]

ATTRIBUTE_AVAILABILITY_VARIETY = [
    "Destruction", "Loss", "Interruption", "Degradation",
    "Acceleration", "Obscuration", "Unknown", "Other",
]

DISCOVERY_METHOD = [
    "Ext - actor disclosure", "Ext - fraud detection", "Ext - monitoring service",
    "Ext - customer", "Ext - unrelated party", "Ext - audit", "Ext - unknown",
    "Int - antivirus", "Int - incident response", "Int - financial audit",
    "Int - fraud detection", "Int - HIDS", "Int - IT audit", "Int - log review",
    "Int - NIDS", "Int - law enforcement", "Int - security alarm", "Int - user",
    "Int - unknown", "Unknown", "Other",
]
