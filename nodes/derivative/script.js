// Дополнительные интерактивные функции для страницы производной
document.addEventListener('DOMContentLoaded', function() {
    // Инициализация всех интерактивных элементов
    initializeDerivativePage();
});

function initializeDerivativePage() {
    // Можно добавить дополнительные интерактивные демонстрации
    console.log('Страница производной загружена');
    
    // Пример: анимация предела
    createLimitAnimation();
}

function createLimitAnimation() {
    // Создание анимации сходящегося предела
    const container = document.createElement('div');
    container.className = 'limit-animation';
    document.querySelector('#definition').appendChild(container);
    
    // Здесь можно добавить D3.js анимацию сходимости
}