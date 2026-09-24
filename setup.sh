#!/bin/bash
# FANet Industrial Anomaly Detection System — Setup Script

set -e

echo "======================================"
echo "  FANet System — Setup Script"
echo "======================================"

# 1. Create virtual environment
echo "[1/9] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
echo "[2/9] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Copy env file
echo "[3/9] Setting up environment file..."
if [ ! -f .env ]; then
  cp .env.example .env
  echo "  Please edit .env and set your MySQL credentials before continuing."
  echo "  Press Enter when ready..."
  read
fi

# 4. Create MySQL database
echo "[4/9] Creating MySQL database..."
DB_NAME=$(grep "^DB_NAME" .env | cut -d= -f2 | tr -d ' ')
DB_USER=$(grep "^DB_USER" .env | cut -d= -f2 | tr -d ' ')
DB_PASSWORD=$(grep "^DB_PASSWORD" .env | cut -d= -f2 | tr -d ' ')
DB_HOST=$(grep "^DB_HOST" .env | cut -d= -f2 | tr -d ' ')
DB_HOST=${DB_HOST:-localhost}

mysql -h "$DB_HOST" -u "$DB_USER" -p"$DB_PASSWORD" \
  -e "CREATE DATABASE IF NOT EXISTS \`${DB_NAME}\` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;" 2>/dev/null \
  && echo "  Database created/confirmed." \
  || echo "  WARNING: Could not auto-create DB. Please run manually:"
  echo "    CREATE DATABASE ${DB_NAME} CHARACTER SET utf8mb4;"

# 5. Run migrations
echo "[5/9] Running database migrations..."
python manage.py makemigrations accounts inspection
python manage.py migrate

# 6. Create default users
echo "[6/9] Creating default users..."
python manage.py create_default_users

# 7. Collect static files
echo "[7/9] Collecting static files..."
python manage.py collectstatic --noinput

# 8. Generate synthetic datasets for all categories
echo "[8/9] Generating synthetic industrial datasets..."
echo "  This creates 300 normal + 50 defective test images per category."
echo "  Categories: bottle, cable, capsule, carpet, grid, leather,"
echo "              metal_nut, tile, transistor, wood, zipper, screw"
python manage.py generate_datasets
echo "  Synthetic datasets generated."

# 9. Pre-train FANet models for all categories
echo "[9/9] Pre-training FANet models..."
echo "  Training may take 5-20 minutes depending on your hardware."
echo "  Skip with Ctrl+C and train later via the web interface."
python manage.py pretrain_models --epochs 10 --backbone efficientnet_b0
echo "  Pre-training complete."

echo ""
echo "======================================"
echo "  Setup Complete!"
echo "======================================"
echo ""
echo "Start the server:"
echo "  source venv/bin/activate"
echo "  python manage.py runserver"
echo ""
echo "Default credentials:"
echo "  Admin:    admin / admin"
echo "  Operator: operator / Operator@123"
echo ""
echo "Access: http://127.0.0.1:8000"
echo ""
echo "To train additional models via CLI:"
echo "  python manage.py train_model --category bottle --epochs 15"
echo ""
echo "To add your own images:"
echo "  Put normal images in:    datasets/<category>/train/good/"
echo "  Put defective images in: datasets/<category>/test/defective/"
echo "  Then retrain:            python manage.py train_model --category <name>"
