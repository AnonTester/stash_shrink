class StashShrinkApp {
    constructor() {
        this.currentResults = [];
        this.selectedScenes = new Set();
        this.currentPage = 1;
        this.pageSize = 50;
        this.sortField = null;
        this.sortDirection = null;
        this.eventSource = null;
        this.isFirstRun = document.body.getAttribute('data-show-settings') === 'True';
        this.handleFirstRun();
        this.queuedSceneIds = new Set();        
        this.totalPages = 1;

        this.initializeTheme();
        this.initializeToastSystem();
        this.initializeEventListeners();
        this.loadConfig();

        // Check if we should show conversion section by default
        this.updateSearchSectionVisibility();
        this.checkInitialView();
    }

    async checkInitialView() {
        // Load initial conversion status to determine what to show
        try {
            const response = await fetch('/api/conversion-status');
            const statusData = await response.json();

            // Track queued scene IDs
            this.updateQueuedSceneIds(statusData.queue);

            const hasQueueItems = statusData.queue && statusData.queue.length > 0;
            
            if (hasQueueItems) {
                this.showConversionSection();
            } else {
                this.showSearchSection();
            }
        } catch (error) {
            console.error('Failed to load initial conversion status:', error);
            this.showSearchSection(); // Fallback to search section
        }
    }

    updateQueuedSceneIds(queue) {
        this.queuedSceneIds.clear();
        queue.forEach(task => this.queuedSceneIds.add(task.scene.id));
        this.renderResults(); // Update checkboxes if results are displayed        
    }        

    initializeToastSystem() {
        this.toastContainer = document.createElement('div');
        this.toastContainer.className = 'toast-container';
        document.body.appendChild(this.toastContainer);
    }

    showToast(message, type = 'info', duration = 5000) {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;

        const icons = {
            success: '✓',
            error: '✕',
            warning: '⚠',
            info: 'ℹ'
        };

        toast.innerHTML = `
            <div class="toast-icon">${icons[type] || icons.info}</div>
            <div class="toast-message">${message}</div>
            <button class="toast-close">&times;</button>
        `;

        this.toastContainer.appendChild(toast);

        // Animate in
        setTimeout(() => toast.classList.add('show'), 10);

        // Close button
        toast.querySelector('.toast-close').addEventListener('click', () => {
            this.hideToast(toast);
        });

        // Auto hide
        if (duration > 0) {
            setTimeout(() => this.hideToast(toast), duration);
        }

        return toast;
    }

    hideToast(toast) {
        toast.classList.remove('show');
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }

    handleFirstRun() {
        if (this.isFirstRun) {
            // Add first-run class to body to dim the background
            document.body.classList.add('first-run');
            console.log('First run detected - settings modal should be open');
        }
    }

    initializeTheme() {
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        document.documentElement.setAttribute('data-theme', prefersDark ? 'dark' : 'light');

        // Listen for theme changes
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
            document.documentElement.setAttribute('data-theme', e.matches ? 'dark' : 'light');
        });
    }

    initializeEventListeners() {
        // Settings modal
        if (!this.isFirstRun) {
            document.getElementById('settings-btn').addEventListener('click', () => this.showSettingsModal());
            document.querySelector('#settings-modal .close').addEventListener('click', () => this.hideSettingsModal());
        }

        // Settings form
        document.getElementById('settings-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.saveSettings(new FormData(e.target));
        });

        // Search form
        document.getElementById('search-form').addEventListener('submit', (e) => this.handleSearch(e));

        // Use video settings button
        document.getElementById('use-video-settings').addEventListener('click', () => this.useVideoSettings());
        
        // Section navigation
        document.getElementById('show-search').addEventListener('click', () => this.showSearchSection());
        document.getElementById('show-conversion').addEventListener('click', () => this.showConversionSection());
        
        // Selection controls
        document.getElementById('select-all').addEventListener('click', () => this.selectAll());
        document.getElementById('select-none').addEventListener('click', () => this.selectNone());
        document.getElementById('select-invert').addEventListener('click', () => this.selectInvert());
        document.getElementById('select-all-checkbox').addEventListener('change', (e) => {
            if (e.target.checked) this.selectAll();
            else this.selectNone();
        });

        // Conversion
        document.getElementById('convert-videos').addEventListener('click', () => this.queueConversion());
        document.getElementById('cancel-all').addEventListener('click', () => this.cancelAllConversions());
        document.getElementById('clear-completed').addEventListener('click', () => this.clearCompleted());

        // Pagination - Top controls
        document.getElementById('page-size-top').addEventListener('change', (e) => {
            this.pageSize = e.target.value === 'all' ? Infinity : parseInt(e.target.value);
            this.currentPage = 1;
            this.syncPaginationControls();
            this.renderResults();
        });

        document.getElementById('first-page-top').addEventListener('click', () => this.goToFirstPage());
        document.getElementById('prev-page-top').addEventListener('click', () => this.previousPage());
        document.getElementById('next-page-top').addEventListener('click', () => this.nextPage());
        document.getElementById('last-page-top').addEventListener('click', () => this.goToLastPage());
        document.getElementById('page-input-top').addEventListener('change', (e) => this.goToPage(parseInt(e.target.value)));
        document.getElementById('page-input-top').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.goToPage(parseInt(e.target.value));
            }
        });

        // Pagination - Bottom controls
        document.getElementById('page-size-bottom').addEventListener('change', (e) => {
            this.pageSize = e.target.value === 'all' ? Infinity : parseInt(e.target.value);
            this.currentPage = 1;
            this.syncPaginationControls();
            this.renderResults();
        });

        document.getElementById('first-page-bottom').addEventListener('click', () => this.goToFirstPage());
        document.getElementById('prev-page-bottom').addEventListener('click', () => this.previousPage());
        document.getElementById('next-page-bottom').addEventListener('click', () => this.nextPage());
        document.getElementById('last-page-bottom').addEventListener('click', () => this.goToLastPage());
        document.getElementById('page-input-bottom').addEventListener('change', (e) => this.goToPage(parseInt(e.target.value)));
        document.getElementById('page-input-bottom').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.goToPage(parseInt(e.target.value));
            }
        });

        // Table sorting - initialize with proper event listeners
        document.querySelectorAll('#results-table th[data-sort-original]').forEach(th => {
            const sortField = th.getAttribute('data-sort-original');
            th.addEventListener('click', () => this.handleSort(sortField));
        });

        // Close modals when clicking outside (only if not first run)
        if (!this.isFirstRun) {
            window.addEventListener('click', (e) => {
                if (e.target.classList.contains('modal')) {
                    this.hideSettingsModal();
                    this.hideLogModal();
                }
            });
        }
        // Prevent escape key on first run
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isFirstRun) {
                e.preventDefault();
                return false;
            }
        });
    }

    // Enhanced pagination methods
    goToFirstPage() {
        if (this.currentPage > 1) {
            this.currentPage = 1;
            this.syncPaginationControls();
            this.renderResults();
            this.updateSearchSectionVisibility();
        }
    }

    goToLastPage() {
        if (this.currentPage < this.totalPages) {
            this.currentPage = this.totalPages;
            this.syncPaginationControls();
            this.renderResults();
        }
    }

    goToPage(page) {
        if (page >= 1 && page <= this.totalPages && page !== this.currentPage) {
            this.currentPage = page;
            this.syncPaginationControls();
            this.renderResults();
        }
    }

    previousPage() {
        if (this.currentPage > 1) {
            this.currentPage--;
            this.syncPaginationControls();
            this.renderResults();
        }
    }

    nextPage() {
        if (this.currentPage < this.totalPages) {
            this.currentPage++;
            this.syncPaginationControls();
            this.renderResults();
        }
    }

    syncPaginationControls() {
        const totalItems = this.currentResults.length;
        this.totalPages = this.pageSize === Infinity ? 1 : Math.ceil(totalItems / this.pageSize);
        
        // Ensure current page is within bounds
        if (this.currentPage > this.totalPages && this.totalPages > 0) {
            this.currentPage = this.totalPages;
        }
        
        // Update page size dropdowns
        document.getElementById('page-size-top').value = this.pageSize === Infinity ? 'all' : this.pageSize.toString();
        document.getElementById('page-size-bottom').value = this.pageSize === Infinity ? 'all' : this.pageSize.toString();
        
        // Update page inputs
        document.getElementById('page-input-top').value = this.currentPage;
        document.getElementById('page-input-bottom').value = this.currentPage;
        
        // Update total pages display
        document.getElementById('total-pages-top').textContent = `of ${this.totalPages}`;
        document.getElementById('total-pages-bottom').textContent = `of ${this.totalPages}`;
        
        // Update page info
        document.getElementById('page-info-top').textContent = `Page ${this.currentPage} of ${this.totalPages}`;
        document.getElementById('page-info-bottom').textContent = `Page ${this.currentPage} of ${this.totalPages}`;
        
        // Update results count
        const resultsText = `${totalItems} result${totalItems !== 1 ? 's' : ''}`;
        document.getElementById('results-count-top').textContent = resultsText;
        document.getElementById('results-count-bottom').textContent = resultsText;
        
        // Update button states
        const firstButtons = document.querySelectorAll('#first-page-top, #first-page-bottom');
        const prevButtons = document.querySelectorAll('#prev-page-top, #prev-page-bottom');
        const nextButtons = document.querySelectorAll('#next-page-top, #next-page-bottom');
        const lastButtons = document.querySelectorAll('#last-page-top, #last-page-bottom');
        
        const isFirstPage = this.currentPage === 1;
        const isLastPage = this.currentPage === this.totalPages || this.pageSize === Infinity;
        
        firstButtons.forEach(btn => btn.disabled = isFirstPage);
        prevButtons.forEach(btn => btn.disabled = isFirstPage);
        nextButtons.forEach(btn => btn.disabled = isLastPage);
        lastButtons.forEach(btn => btn.disabled = isLastPage);
        
        // Update page input bounds
        document.getElementById('page-input-top').max = this.totalPages;
        document.getElementById('page-input-bottom').max = this.totalPages;
    }

    async loadConfig() {
        try {
            const response = await fetch('/api/config');
            const config = await response.json();
            this.config = config;
        } catch (error) {
            console.error('Failed to load config:', error);
        }
    }

    showSettingsModal() {
        const modal = document.getElementById('settings-modal');
        this.populateSettingsForm();
        modal.style.display = 'block';
    }

    hideSettingsModal() {
        // Don't allow hiding during first run
        if (this.isFirstRun) {
            return;
        }
        document.getElementById('settings-modal').style.display = 'none';
    }

    populateSettingsForm() {
        if (!this.config) return;

        const form = document.getElementById('settings-form');
        form.stash_url.value = this.config.stash_url || '';
        form.api_key.value = this.config.api_key || '';
        form.default_search_limit.value = this.config.default_search_limit || 50;
        form.max_concurrent_tasks.value = this.config.max_concurrent_tasks || 2;

        // Populate path mappings
        const pathMappings = this.config.path_mappings || [];
        form.path_mappings.value = pathMappings.join('\n');

        // Populate delete original setting
        const deleteOriginal = this.config.delete_original !== false; // default to true
        form.delete_original.checked = deleteOriginal;

        const videoSettings = this.config.video_settings || {};
        form.width.value = videoSettings.width || '';
        form.height.value = videoSettings.height || '';
        form.bitrate.value = videoSettings.bitrate || '';
        form.framerate.value = videoSettings.framerate || '';
        form.buffer_size.value = videoSettings.buffer_size || '';
        form.container.value = videoSettings.container || '';
    }

    useVideoSettings() {
        if (!this.config || !this.config.video_settings) {
            this.showToast('Video settings not available', 'warning');
            return;
        }
        
        const videoSettings = this.config.video_settings;
        document.getElementById('max_width').value = videoSettings.width || '';
        document.getElementById('max_height').value = videoSettings.height || '';
        document.getElementById('max_bitrate').value = videoSettings.bitrate || '';
        document.getElementById('max_framerate').value = videoSettings.framerate || '';
    }

    async saveSettings(formData) {
        try {
            console.log('Saving settings...');
            
            const settings = {
                stash_url: formData.get('stash_url'),
                api_key: formData.get('api_key'),
                default_search_limit: parseInt(formData.get('default_search_limit')) || 50,
                max_concurrent_tasks: parseInt(formData.get('max_concurrent_tasks')) || 2,
                delete_original: formData.get('delete_original') === 'on',
                path_mappings: formData.get('path_mappings') ? formData.get('path_mappings').split('\n').filter(m => m.trim()) : [],
                video_settings: {
                    width: parseInt(formData.get('width')) || 1280,
                    height: parseInt(formData.get('height')) || 720,
                    bitrate: formData.get('bitrate') || '1000k',
                    framerate: parseFloat(formData.get('framerate')) || 30,
                    buffer_size: formData.get('buffer_size') || '2000k',
                    container: formData.get('container') || 'mp4'
                }
            };
            
            console.log('Sending settings:', settings);
            
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(settings)
            });
            
            if (response.ok) {
                this.config = settings;
                
                if (this.isFirstRun) {
                    this.isFirstRun = false;
                    document.body.classList.remove('first-run');
                    document.getElementById('settings-btn').style.display = 'block';
                    document.getElementById('settings-modal').classList.remove('first-run');
                    document.getElementById('settings-modal').style.display = 'none';
                    
                    this.showToast('Configuration saved successfully! You can now use Stash Shrink.', 'success');
                } else {
                    this.hideSettingsModal();
                    this.showToast('Settings saved successfully!', 'success');
                }
            } else {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to save settings');
            }
        } catch (error) {
            console.error('Error saving settings:', error);
            this.showToast('Error saving settings: ' + error.message, 'error');
        }
    }

    renderResults() {
        const tbody = document.querySelector('#results-table tbody');
        const tableContainer = document.querySelector('.table-container');
        const paginationControls = document.querySelectorAll('.pagination-controls');
        const resultsControls = document.querySelector('.results-controls');        
        
        tbody.innerHTML = '';
        
        if (!this.currentResults || this.currentResults.length === 0) {
            // Hide table and controls when no results
            if (tableContainer) tableContainer.style.display = 'none';
            paginationControls.forEach(control => control.style.display = 'none');
            if (resultsControls) resultsControls.style.display = 'none';
            
            // Show "no results" message
            const noResultsRow = document.createElement('tr');
            noResultsRow.innerHTML = `<td colspan="10" style="text-align: center; padding: 2rem; color: var(--secondary-color);">No scenes found matching your search criteria</td>`;
            tbody.appendChild(noResultsRow);
            this.updateSearchSectionVisibility();
            this.syncPaginationControls();
            return;
        }

        // Show table and controls when there are results
        if (tableContainer) tableContainer.style.display = 'block';
        paginationControls.forEach(control => control.style.display = 'flex');
        if (resultsControls) resultsControls.style.display = 'flex';
        
        let displayResults = [...this.currentResults];
        
        // Apply sorting if active
        if (this.sortField && this.sortDirection) {
            displayResults.sort((a, b) => {
                const aVal = this.getSortValue(a, this.sortField);
                const bVal = this.getSortValue(b, this.sortField);
                
                if (aVal === bVal) return 0;
                
                let result = 0;
                if (typeof aVal === 'string') {
                    result = aVal.localeCompare(bVal);
                } else {
                    result = aVal < bVal ? -1 : 1;
                }
                
                return this.sortDirection === 'desc' ? -result : result;
            });
        }
        
        // Paginate
        const totalItems = displayResults.length;
        const startIndex = this.pageSize === Infinity ? 0 : (this.currentPage - 1) * this.pageSize;
        const endIndex = this.pageSize === Infinity ? totalItems : Math.min(startIndex + this.pageSize, totalItems);
        const pageResults = displayResults.slice(startIndex, endIndex);
        
        pageResults.forEach(scene => {
            const file = scene.files && scene.files.length > 0 ? scene.files[0] : null;
            if (!file) return;
            
            const isQueued = this.queuedSceneIds.has(scene.id);
            const isSelected = this.selectedScenes.has(scene.id) && !isQueued;
            const checkboxDisabled = isQueued ? 'disabled' : '';

            const row = document.createElement('tr');

            row.innerHTML = `
                 <td>
                    <input type="checkbox" class="scene-checkbox" value="${scene.id}" 
                           ${isSelected ? 'checked' : ''} 
                           ${checkboxDisabled}
                           ${isQueued ? 'title="Already in conversion queue"' : ''}>
                    ${isQueued ? '<div style="font-size:10px;color:var(--success-color);">Queued</div>' : ''}
                </td>
                <td class="title-cell" title="${scene.title || 'Untitled'}">
                    <a href="${this.config.stash_url}/scenes/${scene.id}" target="_blank">${scene.title || 'Untitled'}</a>
                </td>
                <td>${this.formatDuration(file.duration)}</td>
                <td>${this.formatFileSize(file.size)}</td>
                <td>${file.video_codec || 'N/A'}</td>
                <td>${file.width || 'N/A'}</td>
                <td>${file.height || 'N/A'}</td>
                <td>${this.formatBitrate(file.bit_rate)}</td>
                <td>${file.frame_rate || 'N/A'}</td>
                <td class="path-cell" title="${file.path}">${this.truncatePath(file.path)}</td>
            `;
            
            row.querySelector('.scene-checkbox').addEventListener('change', (e) => {
                this.toggleSceneSelection(scene.id, e.target.checked);
            });
            
            if (isQueued) row.style.opacity = '0.7';

            tbody.appendChild(row);
        });
        
        this.syncPaginationControls();
        this.updateSelectionControls();
    }

    // Update the search handler to reset pagination
    async handleSearch(e) {
        e.preventDefault();
        
        if (this.isFirstRun) {
            this.showToast('Please complete the first-time setup by saving the configuration.', 'warning');
            return;
        }
        
        const formData = new FormData(e.target);
        
        try {
            const searchParams = {};
            for (let [key, value] of formData.entries()) {
                if (value) searchParams[key] = value;
            }
            
            console.log('Searching with params:', searchParams);
            
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(searchParams)
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            this.currentResults = data.scenes;
            this.currentPage = 1; // Reset to first page on new search
            this.selectedScenes.clear();
            this.syncPaginationControls();
            this.updateSearchSectionVisibility();
            this.renderResults();
            
            this.showToast(`Found ${this.currentResults.length} scenes`, 'success');
            this.updateSearchSectionVisibility();
        } catch (error) {
            console.error('Search failed:', error);
            this.showToast('Search failed: ' + error.message, 'error');
        }
    }

    getSortValue(scene, field) {
        switch (field) {
            case 'title': return scene.title || '';
            case 'duration': return scene.file.duration;
            case 'size': return scene.file.size;
            case 'codec': return scene.file.video_codec;
            case 'width': return scene.file.width;
            case 'height': return scene.file.height;
            case 'bitrate': return scene.file.bit_rate;
            case 'framerate': return scene.file.frame_rate;
            default: return '';
        }
    }

    
    handleSort(field) {
        console.log(`Sorting by ${field}, current field: ${this.sortField}, current direction: ${this.sortDirection}`);
        
        // If clicking the same field, cycle through states
        if (this.sortField === field) {
            if (this.sortDirection === 'asc') {
                this.sortDirection = 'desc';
            } else if (this.sortDirection === 'desc') {
                // Third click: remove sorting
                this.sortField = null;
                this.sortDirection = null;
            }
        } else {
            // New field: start with ascending
            this.sortField = field;
            this.sortDirection = 'asc';
        }
        
        console.log(`New state - field: ${this.sortField}, direction: ${this.sortDirection}`);
        
        this.updateSortIndicators();
        this.renderResults();
    }

    updateSortIndicators() {
        // Remove all sort indicators first
        document.querySelectorAll('#results-table th[data-sort-original]').forEach(th => {
            th.removeAttribute('data-sort');
        });
        
        // Add indicator for current sort field if active
        if (this.sortField && this.sortDirection) {
            const currentTh = document.querySelector(`#results-table th[data-sort-original="${this.sortField}"]`);
            if (currentTh) {
                currentTh.setAttribute('data-sort', this.sortDirection);
            }
        }
    }


    // Update the getSortValue method to handle different data types properly
    getSortValue(scene, field) {
        const file = scene.files && scene.files.length > 0 ? scene.files[0] : null;
        if (!file) return '';
        
        switch (field) {
            case 'title': 
                return scene.title || '';
            case 'duration': 
                return file.duration || 0;
            case 'size': 
                return file.size || 0;
            case 'codec': 
                return file.video_codec || '';
            case 'width': 
                return file.width || 0;
            case 'height': 
                return file.height || 0;
            case 'bitrate': 
                return file.bit_rate || 0;
            case 'framerate': 
                return file.frame_rate || 0;
            default: 
                return '';
        }
    }

    toggleSceneSelection(sceneId, selected) {
        if (selected) {
            this.selectedScenes.add(sceneId);
        } else {
            this.selectedScenes.delete(sceneId);
        }
        this.updateSelectionControls();
    }

    selectAll() {
        // Don't select queued scenes
        const currentPageScenes = this.getCurrentPageSceneIds().filter(id =>
            !this.queuedSceneIds.has(id)
        );
        currentPageScenes.forEach(id => this.selectedScenes.add(id));
        const currentPageScenes = this.getCurrentPageSceneIds();
        currentPageScenes.forEach(id => this.selectedScenes.add(id));
        this.renderResults();
    }

    selectNone() {
        const currentPageScenes = this.getCurrentPageSceneIds();
        currentPageScenes.forEach(id => this.selectedScenes.delete(id));
        this.renderResults();
    }

    selectInvert() {
        const currentPageScenes = this.getCurrentPageSceneIds();
        currentPageScenes.forEach(id => {
            // Skip queued scenes
            if (this.queuedSceneIds.has(id)) return;
            if (this.selectedScenes.has(id)) {
                this.selectedScenes.delete(id);
            } else {
                this.selectedScenes.add(id);
            }
        });
        this.renderResults();
    }

    getCurrentPageSceneIds() {
        let displayResults = [...this.currentResults];

        // Apply sorting
        if (this.sortField && this.sortDirection) {
            displayResults.sort((a, b) => {
                let aVal = this.getSortValue(a, this.sortField);
                let bVal = this.getSortValue(b, this.sortField);

                if (typeof aVal === 'string') {
                    aVal = aVal.toLowerCase();
                    bVal = bVal.toLowerCase();
                }

                if (this.sortDirection === 'asc') {
                    return aVal > bVal ? 1 : aVal < bVal ? -1 : 0;
                } else {
                    return aVal < bVal ? 1 : aVal > bVal ? -1 : 0;
                }
            });
        }

        // Get current page items
        const startIndex = (this.currentPage - 1) * this.pageSize;
        const endIndex = Math.min(startIndex + this.pageSize, displayResults.length);
        const pageResults = this.pageSize === Infinity ? displayResults : displayResults.slice(startIndex, endIndex);

        return pageResults.map(scene => scene.id);
    }

    updateSelectionControls() {
        const currentPageScenes = this.getCurrentPageSceneIds();
        const selectedCount = currentPageScenes.filter(id => this.selectedScenes.has(id)).length;
        const allSelected = selectedCount === currentPageScenes.length;

        document.getElementById('select-all-checkbox').checked = allSelected;
        document.getElementById('select-all-checkbox').indeterminate = selectedCount > 0 && !allSelected;
    }

    updatePagination(totalItems) {
        const totalPages = this.pageSize === Infinity ? 1 : Math.ceil(totalItems / this.pageSize);
        document.getElementById('page-info').textContent = `Page ${this.currentPage} of ${totalPages}`;
        document.getElementById('prev-page').disabled = this.currentPage === 1;
        document.getElementById('next-page').disabled = this.currentPage === totalPages || this.pageSize === Infinity;
    }

    previousPage() {
        if (this.currentPage > 1) {
            this.currentPage--;
            this.renderResults();
        }
    }

    nextPage() {
        const totalPages = Math.ceil(this.currentResults.length / this.pageSize);
        if (this.currentPage < totalPages) {
            this.currentPage++;
            this.renderResults();
        }
    }

    async queueConversion() {
        if (this.isFirstRun) {
            this.showToast('Please complete the first-time setup by saving the configuration.', 'warning');
            return;
        }
        
        if (this.selectedScenes.size === 0) {
            this.showToast('Please select at least one scene to convert.', 'warning');
            return;
        }
        
        try {
            const response = await fetch('/api/queue-conversion', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(Array.from(this.selectedScenes))
            });
            
            if (response.ok) {
                this.showConversionSection();
                this.startSSE();
                this.updateQueuedSceneIds((await response.json()).queue || []);
                this.showToast(`Queued ${this.selectedScenes.size} scenes for conversion.`, 'success');
            } else {
                throw new Error('Failed to queue conversion');
            }
        } catch (error) {
            console.error('Conversion queue failed:', error);
            this.showToast('Failed to queue conversion: ' + error.message, 'error');
        }
    }

    showConversionSection() {
        document.querySelector('.results-section').style.display = 'none';
        document.querySelector('.conversion-section').style.display = 'block';
        document.querySelector('.search-section').style.display = 'none';
        document.getElementById('show-search').style.display = 'inline-block';
        document.getElementById('show-conversion').style.display = 'none';
        this.startSSE(); // Ensure SSE is running when viewing conversions
    }

    showSearchSection() {
        document.querySelector('.results-section').style.display = 'block';
        document.querySelector('.search-section').style.display = 'block';
        document.querySelector('.conversion-section').style.display = 'none';
        document.getElementById('show-search').style.display = 'none';
        document.getElementById('show-conversion').style.display = 'inline-block';        
    }

    startSSE() {
        // Don't start multiple SSE connections
        if (this.eventSource && this.eventSource.readyState !== EventSource.CLOSED) {
            return;
        }
                
        if (this.eventSource) {
            this.eventSource.close();
        }

        this.eventSource = new EventSource('/sse');
        this.eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.updateConversionStatus(data);
        };

        this.eventSource.onerror = (error) => {
            // Don't log normal page unload errors
            if (this.eventSource.readyState === EventSource.CLOSED) {
                console.log('SSE connection closed normally');
            } else {
                console.error('SSE error:', error);
                // Attempt to reconnect after 5 seconds
                setTimeout(() => this.startSSE(), 5000);
            }            
        };
    }

    updateConversionStatus(statusData) {
        this.renderConversionTable(statusData.queue);
        this.updateQueuedSceneIds(statusData.queue);
        this.updateProgressOverview(statusData.queue, statusData.active);
        this.updateConversionUI(statusData.queue);
    }

    updateButtonStates(queue) {
        const hasActiveOrPending = queue.some(task => 
            task.status === 'processing' || task.status === 'pending'
        );
        const hasCompleted = queue.some(task => task.status === 'completed');
        const hasErrors = queue.some(task => task.status === 'error');
        
        // Update Cancel All button
        const cancelAllBtn = document.getElementById('cancel-all');
        if (cancelAllBtn) {
            cancelAllBtn.disabled = !hasActiveOrPending;
        }
        
        // Update Clear Completed button
        const clearCompletedBtn = document.getElementById('clear-completed');
        if (clearCompletedBtn) {
            clearCompletedBtn.disabled = !(hasCompleted || hasErrors);
        }

        return { hasActiveOrPending, hasCompleted, hasErrors, hasAnyTasks: queue.length > 0 };
    }

    updateConversionUI(queue) {
        const buttonStates = this.updateButtonStates(queue);
        const hasAnyTasks = buttonStates.hasAnyTasks;

        // Show/hide conversion section based on whether there are any tasks
        const conversionSection = document.querySelector('.conversion-section');
        if (conversionSection) {
            conversionSection.style.display = hasAnyTasks ? 'block' : 'none';
        }

        this.updateSearchSectionVisibility();
    }
    
    renderConversionTable(queue) {
        const tbody = document.querySelector('#conversion-table tbody');
        tbody.innerHTML = '';

        queue.forEach(task => {
            const row = document.createElement('tr');
            const sceneTitle = task.scene.title || 'Untitled';
            const isError = task.status === 'error';
            const fileName = task.scene.files && task.scene.files.length > 0 ? 
                task.scene.files[0].basename : 'Unknown file';
            const etaText = task.eta && task.eta > 0 ? this.formatTime(task.eta) : 'Calculating...';            
                        
            row.innerHTML = `
                <td class="conversion-title">
                    <div><strong>${sceneTitle}</strong></div>
                    <div style="font-size: 0.875rem; color: var(--secondary-color);">${fileName}</div>
                </td>
                <td class="conversion-status status-${task.status}">${task.status}</td>
                <td class="conversion-progress">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${task.progress}%"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.875rem; margin-top: 0.25rem;">
                        <span>${task.progress.toFixed(1)}%</span>
                        <span style="color: var(--secondary-color);">ETA: ${etaText}</span>
                    </div>
                </td>
                <td class="conversion-actions">
                    ${task.status === 'error' ?
                        `<button class="btn btn-secondary btn-sm" onclick="app.showLog('${task.task_id}')">Log</button>
                         <button class="btn btn-primary btn-sm" onclick="app.retryConversion('${task.task_id}')">Retry</button>` : 
                        ''}
                    ${task.status === 'pending' || task.status === 'processing' ?
                        `<button class="btn btn-danger btn-sm" onclick="app.cancelConversion('${task.task_id}')">Cancel</button>` : 
                        ''}
                    ${task.status === 'completed' ?
                        `<button class="btn btn-secondary btn-sm" onclick="app.removeFromQueue('${task.task_id}')">Remove</button>` : 
                        ''}
                </td>
            `;
            if (isError) row.style.backgroundColor = 'color-mix(in srgb, var(--danger-color) 8%, transparent)';
            tbody.appendChild(row);
        });
    }

    updateProgressOverview(queue, activeTasks) {
        const total = queue.length;
        const completed = queue.filter(task => task.status === 'completed' || task.status === 'error').length;
        const remaining = total - completed;
        const processing = activeTasks ? activeTasks.length : queue.filter(task => task.status === 'processing').length;
        const progress = total > 0 ? (completed / total) * 100 : 0;

        document.getElementById('overall-progress').style.width = `${progress}%`;
        document.getElementById('progress-text').innerHTML = `
            <strong>Total Queue Progress</strong><br>
            ${progress.toFixed(1)}% Complete (${completed}/${total} files, ${remaining} remaining)
        `;

        const activeProcessingTasks = queue.filter(task => task.status === 'processing' && task.eta);
        if (activeProcessingTasks.length > 0) {
            // Use the maximum ETA among active tasks
            const maxEta = Math.max(...activeProcessingTasks.map(task => task.eta || 0));
            document.getElementById('eta-text').textContent = `ETA: ${this.formatTime(maxEta)}`;
        } else if (processing > 0) {
            // Fallback ETA calculation    
            const remaining = total - completed;
            const estimatedTime = remaining * 60; // Placeholder: 1 minute per remaining item
            document.getElementById('eta-text').textContent = `ETA: ${this.formatTime(estimatedTime)}`;
        } else {
            document.getElementById('eta-text').textContent = 'ETA: --';    
        }
    }

    updateSearchSectionVisibility() {
        const searchSection = document.querySelector('.search-section');
        const resultsSection = document.querySelector('.results-section');
        const hasResults = this.currentResults && this.currentResults.length > 0;
        
        if (resultsSection && !this.isFirstRun) {
            resultsSection.style.display = hasResults ? 'block' : 'none';
        }
        
        // Always show search section
    }
    
    async cancelConversion(taskId) {
        try {
            await fetch(`/api/cancel-conversion/${taskId}`, { method: 'POST' });
        } catch (error) {
            console.error('Failed to cancel conversion:', error);
        }
    }

    async cancelAllConversions() {
        if (confirm('Are you sure you want to cancel all conversions?')) {
            try {
                const response = await fetch('/api/cancel-all-conversions', {
                    method: 'POST'
                });
                
                if (response.ok) {
                    this.showToast('All conversions cancelled', 'success');
                } else {
                    throw new Error('Failed to cancel all conversions');
                }
            } catch (error) {
                console.error('Failed to cancel all conversions:', error);
                this.showToast('Failed to cancel all conversions: ' + error.message, 'error');
            }
        }
    }

    async clearCompleted() {
        try {
            await fetch('/api/clear-completed', { method: 'POST' });
        } catch (error) {
            console.error('Failed to clear completed:', error);
        }
    }

    async showLog(taskId) {
        // This would need to fetch the actual log content
        const logContent = "Log content would be displayed here...";
        document.getElementById('log-content').textContent = logContent;
        document.getElementById('log-modal').style.display = 'block';
    }

    hideLogModal() {
        document.getElementById('log-modal').style.display = 'none';
    }

    async retryConversion(taskId) {
        // Implementation would depend on backend API
        alert('Retry functionality to be implemented');
    }

    async removeFromQueue(taskId) {
        // Implementation would depend on backend API
        alert('Remove from queue functionality to be implemented');
    }

    // Utility functions
    formatDuration(seconds) {
        if (!seconds) return '0:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    formatFileSize(bytes) {
        if (!bytes) return '0 B';
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
    }

    formatBitrate(bps) {
        if (!bps) return '0 bps';
        const sizes = ['bps', 'Kbps', 'Mbps', 'Gbps'];
        const i = Math.floor(Math.log(bps) / Math.log(1000));
        return Math.round(bps / Math.pow(1000, i) * 100) / 100 + ' ' + sizes[i];
    }

    formatTime(seconds) {
        if (!seconds || seconds <= 0) return '--';
        
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        if (hours > 0) {
            return `${hours}h ${minutes}m ${secs}s`;
        } else if (minutes > 0) {
            return `${minutes}m ${secs}s`;
        } else {
            return `${secs}s`;
        }        
    }

    truncatePath(path, maxLength = 50) {
        if (path.length <= maxLength) return path;
        return '...' + path.slice(-maxLength + 3);
    }
}

// Log modal close
document.querySelector('#log-modal .close').addEventListener('click', () => {
    app.hideLogModal();
});

// Initialize app when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new StashShrinkApp();
});
