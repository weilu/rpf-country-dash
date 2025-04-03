import bcrypt
import getpass
import random
import string

def generate_random_password(length=10):
    if length < 8:
        raise ValueError("Password length must be at least 8.")

    alphanum_chars = string.ascii_letters + string.digits
    special_char = random.choice(string.punctuation)

    body = ''.join(random.choices(alphanum_chars, k=length - 1))
    return body + special_char


def generate_salted_password():
    password = getpass.getpass("Enter password to hash (leave blank to auto-generate): ")

    if not password:
        password = generate_random_password()
        print(f"🔐 Generated random password: {password}")

    # Convert to bytes
    password_bytes = password.encode('utf-8')

    # Generate salt and hash
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)

    print("✅ Password hash generated successfully!")
    print(f"Plain Password: {password}")
    print(f"Salted Password (store this in DB): {hashed.decode('utf-8')}")

if __name__ == "__main__":
    generate_salted_password()

