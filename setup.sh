cat > setup.sh << 'EOF'
#!/bin/bash
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
EOF