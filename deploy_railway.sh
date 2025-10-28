#!/bin/bash

echo "🚂 Railway 배포 스크립트"
echo "================================"
echo ""

# Railway CLI 로그인 확인
if ! railway whoami &>/dev/null; then
    echo "❌ Railway CLI에 로그인이 필요합니다."
    echo ""
    echo "다음 명령어를 실행하여 로그인하세요:"
    echo "  railway login"
    echo ""
    echo "로그인 후 이 스크립트를 다시 실행하거나 다음 명령어를 실행하세요:"
    echo "  railway init"
    echo "  railway up"
    echo ""
    exit 1
fi

echo "✅ Railway CLI 로그인 확인됨"
echo ""

# 현재 프로젝트 확인
echo "📋 현재 프로젝트 확인..."
railway status

echo ""
echo "🚀 배포를 시작합니다..."
echo ""

# 프로젝트가 이미 연결되어 있는지 확인
if railway status &>/dev/null; then
    echo "📦 기존 프로젝트 연결됨 - 배포 진행..."
    railway up
else
    echo "🆕 새 프로젝트 초기화 중..."
    railway init
    echo "🚀 배포 진행..."
    railway up
fi

echo ""
echo "✅ 배포가 완료되었습니다!"
echo ""
echo "📝 다음 명령어로 로그를 확인하세요:"
echo "  railway logs"
echo ""
echo "🌐 프로젝트를 열려면:"
echo "  railway open"
echo ""

