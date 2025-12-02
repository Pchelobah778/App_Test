document.addEventListener('DOMContentLoaded', function() {
    // Конфигурация
    const config = {
        width: 1200,
        height: 700,
        nodeClickAnimationDuration: 300,
        tooltipDelay: 200
    };

    // Инициализация SVG
    const svg = d3.select('#tree-svg')
        .attr('width', config.width)
        .attr('height', config.height);

    // Создание контейнеров
    const defs = svg.append('defs');
    const linkGroup = svg.append('g').attr('class', 'links');
    const nodeGroup = svg.append('g').attr('class', 'nodes');
    const labelGroup = svg.append('g').attr('class', 'labels');

    // Загрузка данных
    d3.json('data.json').then(data => {
        // Подготавливаем данные
        prepareData(data);
        
        // Рисуем связи
        drawLinks(data);
        
        // Рисуем узлы
        drawNodes(data);
        
        // Добавляем подписи
        drawLabels(data);
        
        // Настраиваем взаимодействие
        setupInteractions(data);
        
        // Настраиваем масштабирование
        setupZoom();
        
    }).catch(error => {
        console.error('Error loading data:', error);
        alert('Ошибка загрузки данных. Проверьте файл data.json');
    });

    function prepareData(data) {
        // Создаем карту узлов по id для быстрого доступа
        data.nodeMap = {};
        data.nodes.forEach(node => {
            data.nodeMap[node.id] = node;
        });
        
        // Добавляем информацию о связях для каждого узла
        data.nodes.forEach(node => {
            node.links = data.links.filter(link => 
                link.source === node.id || link.target === node.id
            );
        });
    }

    // Функция для вычисления точки на кривой SmootherStep
    function smootherStep(t, curvature = 0.3) {
        // SmootherStep функция: 6t^5 - 15t^4 + 10t^3
        const smoother = 6*Math.pow(t,5) - 15*Math.pow(t,4) + 10*Math.pow(t,3);
        // Применяем кривизну для контроля изгиба
        return smoother * curvature;
    }

    function drawLinks(data) {
        // Функция для вычисления пути связи
        function linkPath(d) {
            const source = typeof d.source === 'object' ? d.source : data.nodeMap[d.source];
            const target = typeof d.target === 'object' ? d.target : data.nodeMap[d.target];
            
            if (!source || !target) return '';
            
            const startX = source.x;
            const startY = source.y;
            const endX = target.x;
            const endY = target.y;
            
            // Для прямых линий
            if (d.type === 'straight') {
                return `M${startX},${startY} L${endX},${endY}`;
            }
            
            // Для SmootherStep кривых
            if (d.type === 'smoothstep') {
                const curvature = d.curvature || 0.3;
                const dx = endX - startX;
                const dy = endY - startY;
                
                // Вычисляем контрольные точки для кубической кривой Безье
                // используя SmootherStep для плавного изгиба
                const cp1x = startX + dx * 0.3;
                const cp1y = startY + dy * 0.3 + Math.abs(dx) * curvature * 0.5;
                
                const cp2x = startX + dx * 0.7;
                const cp2y = startY + dy * 0.7 + Math.abs(dx) * curvature * -0.5;
                
                return `M${startX},${startY} C${cp1x},${cp1y} ${cp2x},${cp2y} ${endX},${endY}`;
            }
            
            // По умолчанию - прямая линия
            return `M${startX},${startY} L${endX},${endY}`;
        }
        
        // Рисуем связи
        linkGroup.selectAll('.link')
            .data(data.links)
            .enter()
            .append('path')
            .attr('class', 'link')
            .attr('d', linkPath)
            .attr('stroke', d => d.color || '#95a5a6')
            .attr('stroke-width', d => d.width || 2)
            .attr('fill', 'none')
            .attr('opacity', d => d.opacity || 0.7)
            .attr('stroke-dasharray', d => d.dashed ? '5,3' : null);
    }

    function drawNodes(data) {
        // Рисуем узлы
        nodeGroup.selectAll('.node')
            .data(data.nodes)
            .enter()
            .append('circle')
            .attr('class', 'node')
            .attr('cx', d => d.x)
            .attr('cy', d => d.y)
            .attr('r', d => d.size || 8)
            .attr('fill', d => d.color || '#3498db')
            .attr('stroke', '#fff')
            .attr('stroke-width', 1.5)
            .attr('data-id', d => d.id);
    }

    function drawLabels(data) {
        // Рисуем подписи
        labelGroup.selectAll('.node-label')
            .data(data.nodes)
            .enter()
            .append('text')
            .attr('class', 'node-label')
            .attr('x', d => d.x)
            .attr('y', d => d.y - d.size - 3)
            .attr('text-anchor', 'middle')
            .attr('font-size', d => {
                // Динамический размер шрифта в зависимости от размера узла
                if (d.size > 30) return '14px';
                if (d.size > 20) return '12px';
                return '10px';
            })
            .text(d => d.name);
    }

    function setupInteractions(data) {
        const tooltip = d3.select('#tooltip');
        const nodeInfo = d3.select('#node-info');
        let tooltipTimer;
        
        // Обработчики для узлов
        d3.selectAll('.node')
            .on('mouseover', function(event, d) {
                // Очищаем предыдущий таймер
                clearTimeout(tooltipTimer);
                
                // Подсветка узла
                d3.select(this)
                    .transition()
                    .duration(150)
                    .attr('r', d.size * 1.3)
                    .attr('stroke-width', 2.5);
                
                // Подсветка связанных элементов
                highlightConnections(d.id, true, data);
                
                // Задержка перед показом тултипа
                tooltipTimer = setTimeout(() => {
                    tooltip
                        .style('opacity', 1)
                        .html(`
                            <strong>${d.name}</strong><br>
                            <em>${d.description || 'Нет описания'}</em><br>
                            <small>Кликните для перехода</small>
                        `)
                        .style('left', (event.pageX + 10) + 'px')
                        .style('top', (event.pageY - 10) + 'px');
                }, config.tooltipDelay);
            })
            .on('mouseout', function(event, d) {
                // Очищаем таймер тултипа
                clearTimeout(tooltipTimer);
                
                // Возврат к исходному размеру
                d3.select(this)
                    .transition()
                    .duration(150)
                    .attr('r', d.size)
                    .attr('stroke-width', 1.5);
                
                // Снятие подсветки
                highlightConnections(d.id, false, data);
                
                // Скрытие тултипа
                tooltip.style('opacity', 0);
            })
            .on('click', function(event, d) {
                event.stopPropagation();
                
                // Анимация клика
                const clickAnimation = d3.select(this)
                    .transition()
                    .duration(config.nodeClickAnimationDuration / 2)
                    .attr('r', d.size * 1.5)
                    .transition()
                    .duration(config.nodeClickAnimationDuration / 2)
                    .attr('r', d.size);
                
                // Обновление информации в панели
                updateInfoPanel(d, nodeInfo);
                
                // Переход по ссылке (если она не ссылается на текущую страницу)
                if (d.url && d.url !== '#') {
                    clickAnimation.on('end', () => {
                        window.location.href = d.url;
                    });
                }
            });
        
        // Обработчики для связей
        d3.selectAll('.link')
            .on('mouseover', function(event, d) {
                // Подсветка связи
                d3.select(this)
                    .transition()
                    .duration(150)
                    .attr('stroke-width', (d.width || 2) * 1.5)
                    .attr('opacity', 1);
                
                // Находим связанные узлы
                const source = typeof d.source === 'object' ? d.source : data.nodeMap[d.source];
                const target = typeof d.target === 'object' ? d.target : data.nodeMap[d.target];
                
                if (source) highlightNode(source.id, true);
                if (target) highlightNode(target.id, true);
            })
            .on('mouseout', function(event, d) {
                // Возврат связи к исходному виду
                d3.select(this)
                    .transition()
                    .duration(150)
                    .attr('stroke-width', d.width || 2)
                    .attr('opacity', d.opacity || 0.7);
                
                // Снятие подсветки узлов
                const source = typeof d.source === 'object' ? d.source : data.nodeMap[d.source];
                const target = typeof d.target === 'object' ? d.target : data.nodeMap[d.target];
                
                if (source) highlightNode(source.id, false);
                if (target) highlightNode(target.id, false);
            });
        
        // Обработчики кнопок управления
        d3.select('#reset-view').on('click', () => {
            svg.transition()
                .duration(750)
                .call(zoom.transform, d3.zoomIdentity);
        });
        
        d3.select('#toggle-labels').on('click', () => {
            const labels = d3.selectAll('.node-label');
            const isVisible = labels.style('opacity') !== '0';
            labels
                .transition()
                .duration(300)
                .style('opacity', isVisible ? 0 : 1);
        });
        
        // Клик по пустому месту сбрасывает выделение
        svg.on('click', function(event) {
            if (event.target === this) {
                nodeInfo.html('<p>Кликните на узел для получения информации</p>');
                resetAllHighlights(data);
            }
        });
    }
    
    function updateInfoPanel(node, nodeInfo) {
        const relatedLinks = node.links || [];
        const connections = relatedLinks.map(link => {
            const otherId = link.source === node.id ? link.target : link.source;
            const otherNode = window.treeData?.nodeMap?.[otherId];
            return otherNode ? otherNode.name : otherId;
        }).filter(Boolean);
        
        nodeInfo.html(`
            <h4>${node.name}</h4>
            <p class="description">${node.description || 'Описание отсутствует'}</p>
            <div class="node-details">
                <p><strong>Координаты:</strong> (${Math.round(node.x)}, ${Math.round(node.y)})</p>
                <p><strong>Размер узла:</strong> ${node.size}</p>
                <p><strong>Цвет:</strong> <span style="color:${node.color}">${node.color}</span></p>
                ${connections.length > 0 ? `
                    <p><strong>Связан с:</strong> ${connections.join(', ')}</p>
                ` : ''}
            </div>
            ${node.url && node.url !== '#' ? `
                <a href="${node.url}" class="node-link">
                    Перейти к разделу →
                </a>
            ` : ''}
        `);
    }
    
    function highlightNode(nodeId, highlight) {
        const node = d3.select(`.node[data-id="${nodeId}"]`);
        if (!node.empty()) {
            node
                .transition()
                .duration(150)
                .attr('r', d => highlight ? d.size * 1.4 : d.size)
                .attr('stroke-width', highlight ? 3 : 1.5)
                .attr('filter', highlight ? 'url(#glow)' : null);
        }
    }
    
    function highlightConnections(nodeId, highlight, data) {
        const node = data.nodeMap[nodeId];
        if (!node) return;
        
        // Подсвечиваем связанные связи
        d3.selectAll('.link')
            .attr('opacity', function(link) {
                const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
                const targetId = typeof link.target === 'object' ? link.target.id : link.target;
                
                const isConnected = sourceId === nodeId || targetId === nodeId;
                return highlight ? (isConnected ? 1 : 0.2) : (link.opacity || 0.7);
            })
            .attr('stroke-width', function(link) {
                const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
                const targetId = typeof link.target === 'object' ? link.target.id : link.target;
                
                const isConnected = sourceId === nodeId || targetId === nodeId;
                return highlight ? (isConnected ? (link.width || 2) * 1.5 : link.width || 2) : (link.width || 2);
            });
        
        // Подсвечиваем связанные узлы
        d3.selectAll('.node')
            .attr('opacity', function(d) {
                if (d.id === nodeId) return 1;
                
                const isConnected = node.links.some(link => 
                    link.source === d.id || link.target === d.id
                );
                return highlight ? (isConnected ? 1 : 0.3) : 1;
            });
    }
    
    function resetAllHighlights(data) {
        d3.selectAll('.link')
            .attr('opacity', d => d.opacity || 0.7)
            .attr('stroke-width', d => d.width || 2);
        
        d3.selectAll('.node')
            .attr('opacity', 1)
            .attr('filter', null);
    }
    
    function setupZoom() {
        // Создаем эффект свечения для подсветки
        const filter = defs.append('filter')
            .attr('id', 'glow')
            .attr('height', '300%')
            .attr('width', '300%')
            .attr('x', '-100%')
            .attr('y', '-100%');
            
        filter.append('feGaussianBlur')
            .attr('stdDeviation', '2')
            .attr('result', 'coloredBlur');
            
        const feMerge = filter.append('feMerge');
        feMerge.append('feMergeNode')
            .attr('in', 'coloredBlur');
        feMerge.append('feMergeNode')
            .attr('in', 'SourceGraphic');
        
        // Настраиваем масштабирование
        const zoom = d3.zoom()
            .scaleExtent([0.3, 3])
            .on('zoom', (event) => {
                svg.selectAll('g')
                    .attr('transform', event.transform);
            });
        
        svg.call(zoom);
    }
});