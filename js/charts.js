/**
 * 쿠팡 배송기사 이탈 예측과 리텐션 정책 - 데이터 시각화 스크립트 (Chart.js)
 * 보고서 및 CSV의 실제 수치 100% 반영
 */

// 공통 차트 테마 설정
Chart.defaults.color = '#9DB2D8';
Chart.defaults.font.family = "'Pretendard', sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(7, 18, 43, 0.95)';
Chart.defaults.plugins.tooltip.titleColor = '#5AD1FF';
Chart.defaults.plugins.tooltip.borderColor = 'rgba(90, 209, 255, 0.4)';
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.padding = 12;
Chart.defaults.plugins.tooltip.cornerRadius = 8;

document.addEventListener('DOMContentLoaded', () => {
  initModelComparisonChart();
  initFeatureImportanceChart();
  initHighRiskComparisonChart();
  initWhatIfChart();
  initPolicyRoiChart();
  initMainSummaryChart();
});

/* 1. 머신러닝 모델 성능 비교 차트 (analysis.html) */
function initModelComparisonChart() {
  const ctx = document.getElementById('modelComparisonChart');
  if (!ctx) return;

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Logistic Regression (최종)', 'Random Forest', 'XGBoost', 'Decision Tree'],
      datasets: [
        {
          label: 'Test ROC-AUC',
          data: [0.7830, 0.7703, 0.7626, 0.7341],
          backgroundColor: 'rgba(90, 209, 255, 0.85)',
          borderColor: '#5AD1FF',
          borderWidth: 1.5,
          borderRadius: 6
        },
        {
          label: 'Test PR-AUC',
          data: [0.6264, 0.6069, 0.5821, 0.5135],
          backgroundColor: 'rgba(47, 123, 255, 0.85)',
          borderColor: '#2F7BFF',
          borderWidth: 1.5,
          borderRadius: 6
        },
        {
          label: 'Test Recall (재현율)',
          data: [0.7152, 0.6727, 0.6545, 0.7212],
          backgroundColor: 'rgba(61, 220, 151, 0.8)',
          borderColor: '#3DDC97',
          borderWidth: 1.5,
          borderRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0.4,
          max: 0.85,
          grid: { color: 'rgba(255, 255, 255, 0.08)' },
          ticks: {
            callback: value => value.toFixed(2)
          }
        },
        x: {
          grid: { display: false }
        }
      },
      plugins: {
        legend: {
          position: 'top',
          labels: { boxWidth: 14, font: { weight: 600 } }
        },
        tooltip: {
          callbacks: {
            label: context => ` ${context.dataset.label}: ${(context.raw * 100).toFixed(2)}% (${context.raw})`
          }
        }
      }
    }
  });
}

/* 2. 변수 중요도 Top 10 (Permutation Importance) (analysis.html) */
function initFeatureImportanceChart() {
  const ctx = document.getElementById('featureImportanceChart');
  if (!ctx) return;

  const labels = [
    '1. 근속 (개월)',
    '2. 일평균 근무시간 (h)',
    '3. 직무 만족도 (1~5점)',
    '4. 주당 휴무일 (일)',
    '5. 경로 난이도 (1~10)',
    '6. 12개월 부상 건수',
    '7. 통근 시간 (분)',
    '8. 야간 근무 비중 (0~1)',
    '9. 고용 형태 (위탁 여부)',
    '10. 6개월 관리자 면담'
  ];
  const scores = [0.0335, 0.0287, 0.0236, 0.0185, 0.0175, 0.0133, 0.0129, 0.0115, 0.0077, 0.0075];

  // Top 5는 강조 색상
  const bgColors = scores.map((val, idx) => 
    idx < 5 ? 'rgba(47, 123, 255, 0.9)' : 'rgba(90, 209, 255, 0.4)'
  );

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Permutation Importance 점수',
        data: scores,
        backgroundColor: bgColors,
        borderColor: '#5AD1FF',
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.08)' },
          ticks: {
            callback: value => value.toFixed(3)
          }
        },
        y: {
          grid: { display: false }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: context => ` 중요도 기여도: ${context.raw}`
          }
        }
      }
    }
  });
}

/* 3. 상위 10% 고위험군(300명) vs 일반(2,700명) 특성 비교 (analysis.html) */
function initHighRiskComparisonChart() {
  const ctx = document.getElementById('highRiskComparisonChart');
  if (!ctx) return;

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: [
        '위탁 기사 비율',
        '면담 1회 이하 비율',
        '만족도 3점 미만 비율',
        '주당 휴무 1일 이하',
        '근속 6개월 이하 비율'
      ],
      datasets: [
        {
          label: '상위 10% 고위험군 (300명)',
          data: [81.3, 73.7, 79.7, 67.0, 58.7],
          backgroundColor: 'rgba(255, 122, 69, 0.85)',
          borderColor: '#FF7A45',
          borderWidth: 1.5,
          borderRadius: 6
        },
        {
          label: '일반 기사 (2,700명)',
          data: [51.9, 37.9, 22.0, 24.8, 26.8],
          backgroundColor: 'rgba(90, 209, 255, 0.45)',
          borderColor: '#5AD1FF',
          borderWidth: 1.5,
          borderRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          max: 100,
          grid: { color: 'rgba(255, 255, 255, 0.08)' },
          ticks: { callback: value => `${value}%` }
        },
        x: { grid: { display: false } }
      },
      plugins: {
        legend: { position: 'top' },
        tooltip: {
          callbacks: {
            label: context => ` ${context.dataset.label}: ${context.raw}%`
          }
        }
      }
    }
  });
}

/* 4. What-if 정책 시나리오별 평균 예측 이탈률 (policy.html) */
function initWhatIfChart() {
  const ctx = document.getElementById('whatIfChart');
  if (!ctx) return;

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['기준선 (현재)', '면담 +2회 (C)', '근무 -1시간 (B)', '휴무 +1일 (A)', '복합 정책 (D)'],
      datasets: [{
        label: '평균 예측 이탈률 (%)',
        data: [44.84, 40.37, 40.04, 37.87, 24.78],
        backgroundColor: [
          'rgba(255, 122, 69, 0.85)',
          'rgba(61, 220, 151, 0.85)', // 유일한 양의 ROI 강조
          'rgba(90, 209, 255, 0.65)',
          'rgba(47, 123, 255, 0.65)',
          'rgba(147, 51, 234, 0.75)'
        ],
        borderWidth: 1.5,
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0,
          max: 50,
          grid: { color: 'rgba(255, 255, 255, 0.08)' },
          ticks: { callback: value => `${value}%` }
        },
        x: { grid: { display: false } }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: context => ` 예측 이탈률: ${context.raw}% (기준 대비 ${(44.84 - context.raw).toFixed(2)}%p 감소)`
          }
        }
      }
    }
  });
}

/* 5. 리텐션 정책 순효과 및 추정 ROI (policy.html) */
function initPolicyRoiChart() {
  const ctx = document.getElementById('policyRoiChart');
  if (!ctx) return;

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['면담 +2회 (C)', '복합 정책 (D)', '근무 -1h (B)', '휴무 +1일 (A)', '소득 +10% (참고)'],
      datasets: [{
        label: '추정 ROI (%)',
        data: [78.8, -76.38, -84.0, -86.74, -96.6],
        backgroundColor: [
          'rgba(61, 220, 151, 0.9)', // +78.8%
          'rgba(255, 122, 69, 0.75)',
          'rgba(255, 122, 69, 0.75)',
          'rgba(255, 122, 69, 0.75)',
          'rgba(239, 68, 68, 0.9)'
        ],
        borderWidth: 1.5,
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.08)' },
          ticks: { callback: value => `${value}%` }
        },
        x: { grid: { display: false } }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: context => ` 추정 ROI: ${context.raw > 0 ? '+' : ''}${context.raw}%`
          }
        }
      }
    }
  });
}

/* 6. 메인 요약 차트 (index.html) */
function initMainSummaryChart() {
  const ctx = document.getElementById('mainSummaryChart');
  if (!ctx) return;

  new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['재직 유지 (72.47%)', '6개월 내 이탈 (27.53%)'],
      datasets: [{
        data: [2174, 826],
        backgroundColor: [
          'rgba(47, 123, 255, 0.85)',
          'rgba(255, 122, 69, 0.9)'
        ],
        borderColor: '#07122B',
        borderWidth: 3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { boxWidth: 12, padding: 16 }
        },
        tooltip: {
          callbacks: {
            label: context => ` ${context.label}: ${context.raw}명`
          }
        }
      },
      cutout: '68%'
    }
  });
}
