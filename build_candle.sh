#!/bin/bash

# Candle 版本构建和运行脚本

echo "Building Candle version of MyGPT..."

# 备份原始 Cargo.toml
if [ -f "Cargo.toml" ]; then
    echo "Backing up original Cargo.toml to Cargo_burn.toml.bak"
    cp Cargo.toml Cargo_burn.toml.bak
fi

# 使用 Candle 版本的 Cargo.toml
echo "Using Cargo_candle.toml..."
cp Cargo_candle.toml Cargo.toml

# 清理之前的构建
echo "Cleaning previous builds..."
cargo clean

# 构建项目
echo "Building project..."
cargo build --release --bin mygpt-candle

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo ""
    echo "Running mygpt-candle..."
    echo "========================================"
    cargo run --release --bin mygpt-candle
else
    echo "Build failed!"
    # 恢复原始 Cargo.toml
    if [ -f "Cargo_burn.toml.bak" ]; then
        cp Cargo_burn.toml.bak Cargo.toml
        rm Cargo_burn.toml.bak
    fi
    exit 1
fi

# 恢复原始 Cargo.toml
if [ -f "Cargo_burn.toml.bak" ]; then
    echo ""
    echo "Restoring original Cargo.toml..."
    cp Cargo_burn.toml.bak Cargo.toml
    rm Cargo_burn.toml.bak
fi

echo "Done!"
