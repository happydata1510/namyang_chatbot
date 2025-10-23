// 카드 선택 기능
document.querySelectorAll('.card').forEach(card => {
  card.addEventListener('click', function() {
    document.querySelectorAll('.card').forEach(c => c.classList.remove('selected'));
    this.classList.add('selected');
  });
});

// 페이지 로드 시 실행
document.addEventListener('DOMContentLoaded', function() {
  // 현재 페이지에 맞는 옵션 선택
  const select = document.getElementById('violence-select');
  const currentPath = window.location.pathname;
  const currentPage = currentPath.substring(currentPath.lastIndexOf('/') + 1) || 'index.html';
  
  // 페이지 이름과 선택박스 value 매핑
  const pageMapping = {
    'index.html': '',
    'common.html': 'common.html',
    'child.html': 'child.html',
    'family.html': 'family.html',
    'elder.html': 'elder.html',
    'school.html': 'school.html',
    'sex.html': 'sex.html',
    'stalking.html': 'stalking.html'
  };

  // 현재 페이지에 맞는 옵션 선택
  if (pageMapping.hasOwnProperty(currentPage)) {
    select.value = pageMapping[currentPage];
  } else {
    select.value = '';
  }

  // 페이지 이동 이벤트
  if (select) {
    select.addEventListener('change', function() {
      if (this.value) {
        document.body.classList.add('fade-out');
        setTimeout(() => {
          window.location.href = this.value;
        }, 500); // 0.5초 후 이동
      }
    });
  }

  // 모든 슬라이더 초기화
  document.querySelectorAll('.restorative-slider').forEach(slider => {
    const images = slider.querySelectorAll('.slider-images img');
    const prevBtn = slider.querySelector('.slider-btn.prev');
    const nextBtn = slider.querySelector('.slider-btn.next');
    const indicator = slider.querySelector('.slider-indicator');
    
    if (images.length) {
      let current = 0;
      let autoSlideInterval = null;

      function renderIndicator(idx) {
        if (!indicator) return;
        indicator.innerHTML = '';
        images.forEach((img, i) => {
          const dot = document.createElement('span');
          if (i === idx) dot.classList.add('active');
          indicator.appendChild(dot);
        });
      }

      function showImage(idx) {
        images.forEach((img, i) => {
          img.classList.toggle('active', i === idx);
        });
        renderIndicator(idx);
      }

      function nextSlide() {
        current = (current + 1) % images.length;
        showImage(current);
      }

      function startAutoSlide() {
        // 기존 인터벌이 있다면 제거
        if (autoSlideInterval) {
          clearInterval(autoSlideInterval);
        }
        // 3초마다 자동으로 다음 슬라이드로 이동
        autoSlideInterval = setInterval(nextSlide, 3000);
      }

      function stopAutoSlide() {
        if (autoSlideInterval) {
          clearInterval(autoSlideInterval);
          autoSlideInterval = null;
        }
      }

      showImage(current);

      // 스토킹 페이지에만 자동 슬라이드 기능 적용
      const isStalkingPage = window.location.pathname.includes('stalking') || 
                            window.location.href.includes('stalking');
      
      if (isStalkingPage) {
        startAutoSlide(); // 스토킹 페이지에서만 자동 슬라이드 시작
      }

      if (prevBtn && nextBtn) {
        prevBtn.addEventListener('click', () => {
          current = (current - 1 + images.length) % images.length;
          showImage(current);
          // 수동 조작 시 자동 슬라이드 재시작 (스토킹 페이지에서만)
          if (isStalkingPage) {
            stopAutoSlide();
            startAutoSlide();
          }
        });
        nextBtn.addEventListener('click', () => {
          current = (current + 1) % images.length;
          showImage(current);
          // 수동 조작 시 자동 슬라이드 재시작 (스토킹 페이지에서만)
          if (isStalkingPage) {
            stopAutoSlide();
            startAutoSlide();
          }
        });
      }

      // 이미지 클릭 시에도 자동 슬라이드 재시작 (스토킹 페이지에서만)
      if (isStalkingPage) {
        images.forEach((img, index) => {
          img.addEventListener('click', () => {
            current = index;
            showImage(current);
            stopAutoSlide();
            startAutoSlide();
          });
        });
      }

      // 슬라이더에 마우스가 올라가면 자동 슬라이드 일시정지 (스토킹 페이지에서만)
      if (isStalkingPage) {
        slider.addEventListener('mouseenter', stopAutoSlide);
        slider.addEventListener('mouseleave', startAutoSlide);
      }
    }
  });

  // 맨 위로 버튼 동작
  const scrollTopBtn = document.getElementById('scrollTopBtn');
  if (scrollTopBtn) {
    window.addEventListener('scroll', function() {
      if (document.documentElement.scrollTop > 200 || document.body.scrollTop > 200) {
        scrollTopBtn.classList.add('show');
      } else {
        scrollTopBtn.classList.remove('show');
      }
    });
    scrollTopBtn.onclick = function() {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    };
  }
}); 