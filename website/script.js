// script.js

// Smooth scrolling for in-page anchors
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', e => {
    e.preventDefault();
    const target = document.querySelector(anchor.getAttribute('href'));
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});

// Optional: scroll-to-top button
const toTopBtn = document.createElement('button');
toTopBtn.textContent = "↑ Top";
Object.assign(toTopBtn.style, {
  position: 'fixed',
  bottom: '20px',
  right: '20px',
  display: 'none',
  padding: '8px 12px',
  'font-size': '16px',
  'border-radius': '4px',
  border: 'none',
  background: '#0078d4',
  color: '#fff',
  cursor: 'pointer',
  'z-index': 1000
});
document.body.appendChild(toTopBtn);

window.addEventListener('scroll', () => {
  toTopBtn.style.display = window.scrollY > 300 ? 'block' : 'none';
});
toTopBtn.addEventListener('click', () => {
  window.scrollTo({ top: 0, behavior: 'smooth' });
});
