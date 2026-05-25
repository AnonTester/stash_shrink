class StashShrinkApp {
    constructor() {
        this.currentResults = [];
        this.selectedScenes = new Set();
        this.currentPage = 1;
        this.pageSize = 50;
        this.sortField = null;
        this.sortDirection = null;
        this.eventSource = null;
        this.lastConversionStatus = null; // Cache last conversion status
        this.isFirstRun = document.body.getAttribute('data-show-settings') === 'True';
        this.handleFirstRun();
        this.queuedSceneIds = new Set();
        this.isQueuePaused = true; // Runtime-only state, default to paused
        this.totalPages = 1;
        this.updateStatus = null;
        this.lastUpdatePromptedVersion = null;
        this.endpointConfigs = [];
        this.settingsEndpointId = null;
        this.searchEndpointId = null;
        this.currentResultsEndpointId = null;

        // Store section references
        this.searchSection = document.querySelector('.search-section');
        this.resultsSection = document.querySelector('.results-section');
        this.conversionSection = document.querySelector('.conversion-section');
        this.showSearchBtn = document.getElementById('show-search');
        this.showConversionBtn = document.getElementById('show-conversion');

        // Store conversion control references
        this.conversionControls = document.querySelector('.conversion-controls');
        this.progressOverview = document.querySelector('.progress-overview');

        this.initializeTheme();
        this.initializeToastSystem();
        this.initializeEventListeners();
        this.loadConfig();

         // Add page visibility listener
        this.setupVisibilityListener();

        this.checkInitialView();
    }

    setupVisibilityListener() {
        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseSSE();
            } else {
                this.resumeSSE();
            }
        });
    }

    pauseSSE() {
        if (this.eventSource) {
            console.log('Pausing SSE due to page visibility change');
            this.eventSource.close();
            this.eventSource = null;
        }
    }

    resumeSSE() {
        // Only resume if we're in the conversion section or have active tasks
        if (this.conversionSection && this.conversionSection.style.display === 'block') {
            console.log('Resuming SSE after page visibility restored');
            // Start SSE without blocking UI
            this.startSSE().catch(error => {
                console.error('Failed to resume SSE:', error);
            });
        }
    }

    async checkInitialView() {
        // Load initial conversion status to determine what to show
        try {
            const response = await fetch('/api/conversion-status');
            const statusData = await response.json();
            this.updateConversionStatus(statusData);

            const hasQueueItems = statusData.queue && statusData.queue.length > 0;

            // Always start with search section
            this.showSearchSection();

            if (hasQueueItems) {
                this.showConversionBtn.style.display = 'inline-block';
            }
        } catch (error) {
            console.error('Failed to load initial conversion status:', error);
            this.showSearchSection(); // Fallback to search section
        }
    }

    updateQueuedSceneIds(queue) {
        // Only update if there's an actual change
        const newIds = new Set(queue ? queue.map(task => {
            const endpointId = task.endpoint_id || this.config?.active_endpoint_id;
            return this.getQueueSceneKey(task.scene.id, endpointId);
        }) : []);
        if (this.setsAreEqual(this.queuedSceneIds, newIds)) {
            return; // No change, don't re-render
        }

        this.queuedSceneIds = newIds;
        // Only render results if they're currently displayed
        if (this.resultsSection.style.display !== 'none') {
            this.renderResults();
        }
    }

    setsAreEqual(set1, set2) {
        if (set1.size !== set2.size) return false;
        for (let item of set1) {
            if (!set2.has(item)) return false;
        }
        return true;
    }

    getDefaultVideoSettings() {
        return {
            width: 1280,
            height: 720,
            bitrate: '1000k',
            framerate: 30,
            min_filesize: '',
            buffer_size: '2000k',
            container: 'mp4',
            crf: 26
        };
    }

    generateEndpointId() {
        if (window.crypto && crypto.randomUUID) {
            return crypto.randomUUID();
        }
        return `endpoint-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }

    createEndpointConfig(name) {
        return {
            id: this.generateEndpointId(),
            name: name || 'Endpoint',
            stash_url: '',
            api_key: '',
            video_settings: { ...this.getDefaultVideoSettings() },
            path_mappings: []
        };
    }

    normalizeEndpointConfig(endpoint, index) {
        const normalized = { ...endpoint };
        if (!normalized.id) {
            normalized.id = this.generateEndpointId();
        }
        if (!normalized.name) {
            normalized.name = index === 0 ? 'default' : `Endpoint ${index + 1}`;
        }
        normalized.stash_url = normalized.stash_url || '';
        normalized.api_key = normalized.api_key || '';
        normalized.video_settings = {
            ...this.getDefaultVideoSettings(),
            ...(normalized.video_settings || {})
        };
        normalized.path_mappings = Array.isArray(normalized.path_mappings) ? normalized.path_mappings : [];
        return normalized;
    }

    initializeEndpointState() {
        if (!this.config) return;

        let endpoints = [];
        if (Array.isArray(this.config.endpoints)) {
            endpoints = this.config.endpoints;
        } else if (this.config.stash_url) {
            endpoints = [{
                id: 'default',
                name: 'default',
                stash_url: this.config.stash_url,
                api_key: this.config.api_key || '',
                video_settings: this.config.video_settings || {},
                path_mappings: this.config.path_mappings || []
            }];
        }

        this.endpointConfigs = endpoints.map((endpoint, index) => this.normalizeEndpointConfig(endpoint, index));

        if (this.endpointConfigs.length === 0) {
            const defaultEndpoint = this.createEndpointConfig('default');
            defaultEndpoint.name = 'default';
            this.endpointConfigs = [this.normalizeEndpointConfig(defaultEndpoint, 0)];
        }

        this.config.endpoints = this.endpointConfigs;

        const activeId = this.config.active_endpoint_id;
        if (!activeId || !this.getEndpointById(activeId)) {
            this.config.active_endpoint_id = this.endpointConfigs[0].id;
        }

        const normalizedActiveId = this.config.active_endpoint_id;
        if (!this.settingsEndpointId || !this.getEndpointById(this.settingsEndpointId)) {
            this.settingsEndpointId = normalizedActiveId;
        }

        if (!this.searchEndpointId || !this.getEndpointById(this.searchEndpointId)) {
            this.searchEndpointId = normalizedActiveId;
        }

        if (!this.currentResultsEndpointId || !this.getEndpointById(this.currentResultsEndpointId)) {
            this.currentResultsEndpointId = this.searchEndpointId;
        }

        this.renderSearchEndpointOptions();
    }

    getEndpointById(endpointId) {
        if (!this.endpointConfigs) return null;
        return this.endpointConfigs.find(endpoint => endpoint.id === endpointId) || null;
    }

    renderEndpointSelectOptions() {
        const select = document.getElementById('endpoint-config-select');
        if (!select) return;

        select.innerHTML = '';
        this.endpointConfigs.forEach(endpoint => {
            const option = document.createElement('option');
            option.value = endpoint.id;
            option.textContent = endpoint.name || endpoint.stash_url || 'Endpoint';
            select.appendChild(option);
        });

        if (!this.settingsEndpointId || !this.getEndpointById(this.settingsEndpointId)) {
            this.settingsEndpointId = this.endpointConfigs[0]?.id || null;
        }

        select.value = this.settingsEndpointId || '';
        this.updateEndpointActionState();
    }

    renderSearchEndpointOptions() {
        const select = document.getElementById('search-endpoint');
        const container = document.getElementById('search-endpoint-container');
        if (!select || !container) return;

        select.innerHTML = '';
        this.endpointConfigs.forEach(endpoint => {
            const option = document.createElement('option');
            option.value = endpoint.id;
            option.textContent = endpoint.name || endpoint.stash_url || 'Endpoint';
            select.appendChild(option);
        });

        const showSelector = this.endpointConfigs.length > 1;
        container.style.display = showSelector ? 'flex' : 'none';

        if (!this.searchEndpointId || !this.getEndpointById(this.searchEndpointId)) {
            this.searchEndpointId = this.config?.active_endpoint_id || this.endpointConfigs[0]?.id || null;
        }

        select.value = this.searchEndpointId || '';
    }

    updateEndpointActionState() {
        const removeButton = document.getElementById('remove-endpoint');
        if (!removeButton) return;
        removeButton.disabled = this.endpointConfigs.length <= 1;
    }

    populateEndpointForm() {
        const endpoint = this.getEndpointById(this.settingsEndpointId);
        if (!endpoint) return;

        const form = document.getElementById('settings-form');
        form.endpoint_name.value = endpoint.name || '';
        form.stash_url.value = endpoint.stash_url || '';
        form.api_key.value = endpoint.api_key || '';

        const videoSettings = endpoint.video_settings || {};
        form.width.value = videoSettings.width || '';
        form.height.value = videoSettings.height || '';
        form.bitrate.value = videoSettings.bitrate || '';
        form.framerate.value = videoSettings.framerate || '';
        form.settings_min_filesize.value = videoSettings.min_filesize || '';
        form.buffer_size.value = videoSettings.buffer_size || '';
        form.container.value = videoSettings.container || '';
        form.crf.value = videoSettings.crf || 26;
        document.getElementById('crf-value').textContent = videoSettings.crf || 26;

        const pathMappings = endpoint.path_mappings || [];
        form.path_mappings.value = pathMappings.join('\n');
    }

    persistEndpointForm() {
        const endpoint = this.getEndpointById(this.settingsEndpointId);
        if (!endpoint) return;

        const form = document.getElementById('settings-form');
        endpoint.name = (form.endpoint_name.value || '').trim() || endpoint.name || 'default';
        endpoint.stash_url = (form.stash_url.value || '').trim();
        endpoint.api_key = (form.api_key.value || '').trim();
        endpoint.path_mappings = form.path_mappings.value
            ? form.path_mappings.value.split('\n').filter(mapping => mapping.trim())
            : [];
        endpoint.video_settings = {
            width: parseInt(form.width.value) || 1280,
            height: parseInt(form.height.value) || 720,
            bitrate: form.bitrate.value || '1000k',
            framerate: parseFloat(form.framerate.value) || 30,
            min_filesize: form.settings_min_filesize.value || '',
            buffer_size: form.buffer_size.value || '2000k',
            container: form.container.value || 'mp4',
            crf: parseInt(form.crf.value) || 26
        };
    }

    getQueueSceneKey(sceneId, endpointId) {
        return `${endpointId || 'default'}:${sceneId}`;
    }

    addEndpoint() {
        this.persistEndpointForm();

        const existingNames = new Set(this.endpointConfigs.map(endpoint => (endpoint.name || '').toLowerCase()));
        let index = this.endpointConfigs.length + 1;
        let name = `Endpoint ${index}`;
        while (existingNames.has(name.toLowerCase())) {
            index += 1;
            name = `Endpoint ${index}`;
        }

        const newEndpoint = this.createEndpointConfig(name);
        this.endpointConfigs.push(newEndpoint);
        this.settingsEndpointId = newEndpoint.id;
        this.renderEndpointSelectOptions();
        this.populateEndpointForm();
        this.renderSearchEndpointOptions();
    }

    removeEndpoint() {
        if (this.endpointConfigs.length <= 1) {
            this.showToast('At least one endpoint is required.', 'warning');
            return;
        }

        const removeIndex = this.endpointConfigs.findIndex(endpoint => endpoint.id === this.settingsEndpointId);
        if (removeIndex === -1) return;

        const removedEndpoint = this.endpointConfigs.splice(removeIndex, 1)[0];
        const fallbackId = this.endpointConfigs[0]?.id || null;

        this.settingsEndpointId = fallbackId;
        if (this.searchEndpointId === removedEndpoint.id) {
            this.searchEndpointId = fallbackId;
        }
        if (this.currentResultsEndpointId === removedEndpoint.id) {
            this.currentResultsEndpointId = fallbackId;
        }

        this.renderEndpointSelectOptions();
        this.populateEndpointForm();
        this.renderSearchEndpointOptions();
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

    async extractErrorMessage(response) {
        try {
            const cloned = response.clone();
            const data = await cloned.json();

            if (data) {
                if (typeof data.detail === 'string') return data.detail;
                if (Array.isArray(data.detail)) return data.detail.join(', ');
                if (data.errors) {
                    const errorMessages = Array.isArray(data.errors) ? data.errors : [data.errors];
                    return errorMessages.map(err => err.message || err).join(', ');
                }
            }
        } catch (jsonError) {
            console.warn('Failed to parse error JSON response', jsonError);
        }

        try {
            return await response.text();
        } catch (textError) {
            console.warn('Failed to read error response text', textError);
            return '';
        }
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

        const checkUpdatesBtn = document.getElementById('check-updates-btn');
        if (checkUpdatesBtn) {
            checkUpdatesBtn.addEventListener('click', () => this.manualUpdateCheck());
        }

        // Update modal
        this.updateModal = document.getElementById('update-modal');
        const updateClose = document.querySelector('#update-modal .close');
        if (updateClose) {
            updateClose.addEventListener('click', () => this.hideUpdateModal());
        }
        const closeUpdateBtn = document.getElementById('close-update-btn');
        if (closeUpdateBtn) {
            closeUpdateBtn.addEventListener('click', () => this.hideUpdateModal());
        }
        const applyUpdateBtn = document.getElementById('apply-update-btn');
        if (applyUpdateBtn) {
            applyUpdateBtn.addEventListener('click', () => this.applyUpdate());
        }

        // Settings form
        document.getElementById('settings-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.saveSettings(new FormData(e.target));
        });

        const endpointSelect = document.getElementById('endpoint-config-select');
        if (endpointSelect) {
            endpointSelect.addEventListener('change', (e) => {
                this.persistEndpointForm();
                this.settingsEndpointId = e.target.value;
                this.populateEndpointForm();
                this.renderSearchEndpointOptions();
            });
        }

        const endpointNameInput = document.getElementById('endpoint_name');
        if (endpointNameInput) {
            endpointNameInput.addEventListener('input', (e) => {
                const endpoint = this.getEndpointById(this.settingsEndpointId);
                if (!endpoint) return;
                endpoint.name = e.target.value;
                this.renderEndpointSelectOptions();
                this.renderSearchEndpointOptions();
            });
        }

        const addEndpointButton = document.getElementById('add-endpoint');
        if (addEndpointButton) {
            addEndpointButton.addEventListener('click', () => this.addEndpoint());
        }

        const removeEndpointButton = document.getElementById('remove-endpoint');
        if (removeEndpointButton) {
            removeEndpointButton.addEventListener('click', () => this.removeEndpoint());
        }

        // CRF slider value display
        const crfSlider = document.getElementById('crf');
        const crfValue = document.getElementById('crf-value');
        if (crfSlider && crfValue) {
            crfSlider.addEventListener('input', (e) => {
                crfValue.textContent = e.target.value;
            });
        }

        const searchEndpointSelect = document.getElementById('search-endpoint');
        if (searchEndpointSelect) {
            searchEndpointSelect.addEventListener('change', (e) => {
                this.searchEndpointId = e.target.value;
            });
        }

        // Search form
        document.getElementById('search-form').addEventListener('submit', (e) => this.handleSearch(e));

        // Use video settings button
        document.getElementById('use-video-settings').addEventListener('click', () => this.useVideoSettings());

        // Section navigation
        this.showSearchBtn.addEventListener('click', () => this.showSearchSection());
        this.showConversionBtn.addEventListener('click', () => this.showConversionSection());

        // Selection controls
        document.getElementById('select-all').addEventListener('click', () => this.selectAll());
        document.getElementById('select-none').addEventListener('click', () => this.selectNone());
        document.getElementById('select-invert').addEventListener('click', () => this.selectInvert());
        document.getElementById('select-all-checkbox').addEventListener('change', (e) => {
            const currentPageScenes = this.getCurrentPageSceneIds();
            const endpointId = this.currentResultsEndpointId || this.searchEndpointId;
            const selectableSceneIds = currentPageScenes.filter(
                id => !this.queuedSceneIds.has(this.getQueueSceneKey(id, endpointId))
            );

            if (e.target.checked && selectableSceneIds.length > 0) {
                this.selectAll();
            } else {
                this.selectNone();
            }
        });

        // Conversion
        document.getElementById('convert-videos').addEventListener('click', () => this.queueConversion());
        document.getElementById('cancel-all').addEventListener('click', () => this.cancelAllConversions());
        document.getElementById('clear-completed').addEventListener('click', () => this.clearCompleted());
        document.getElementById('clear-errors').addEventListener('click', () => this.clearErrors());
        document.getElementById('retry-all-errors').addEventListener('click', () => this.retryAllErrors());
        document.getElementById('toggle-pause').addEventListener('click', () => this.toggleQueuePause());
        document.getElementById('remove-all-pending').addEventListener('click', () => this.removeAllPending());

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
                    this.hideUpdateModal();
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

        // Event delegation for conversion table buttons
        document.querySelector('#conversion-table').addEventListener('click', (e) => {
            const button = e.target.closest('button');
            if (!button) return;

            const taskId = button.getAttribute('data-task-id');
            const action = button.getAttribute('data-action');

            if (action === 'cancel' && taskId) this.cancelConversion(taskId);
            if (action === 'remove' && taskId) this.removeFromQueue(taskId);
            if (action === 'show-log' && taskId) this.showLog(taskId);
            if (action === 'retry' && taskId) this.retryConversion(taskId);
            if (action === 'retry-stash' && taskId) this.retryStashFix(taskId);
        });
    }

    // Enhanced pagination methods
    goToFirstPage() {
        if (this.currentPage > 1) {
            this.currentPage = 1;
            this.syncPaginationControls();
            this.renderResults();
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
            this.initializeEndpointState();
        } catch (error) {
            console.error('Failed to load config:', error);
        }
    }

    showSettingsModal() {
        const modal = document.getElementById('settings-modal');
        this.populateSettingsForm();
        modal.style.display = 'block';

        // Prevent background scrolling when modal is open
        document.body.style.overflow = 'hidden';
        document.body.style.position = 'fixed';
        document.body.style.width = '100%';
        document.body.style.height = '100%';
        document.body.style.top = '0';
        document.body.style.left = '0';
    }

    hideSettingsModal() {
        // Don't allow hiding during first run
        if (this.isFirstRun) {
            return;
        }
        document.getElementById('settings-modal').style.display = 'none';

        // Restore background scrolling when modal is closed
        document.body.style.overflow = '';
        document.body.style.position = '';
        document.body.style.width = '';
        document.body.style.height = '';
        document.body.style.top = '';
        document.body.style.left = '';
    }

    showUpdateModal() {
        if (!this.updateModal) return;
        this.updateModal.style.display = 'block';
        this.updateModal.style.zIndex = '1100';
        const settingsModal = document.getElementById('settings-modal');
        if (settingsModal) {
            settingsModal.style.zIndex = '1000';
        }
        document.body.style.overflow = 'hidden';
    }

    hideUpdateModal() {
        if (!this.updateModal) return;
        this.updateModal.style.display = 'none';
        document.body.style.overflow = '';
    }

    populateUpdateModal(updateInfo) {
        if (!updateInfo) return;
        const currentEl = document.getElementById('current-version');
        const latestEl = document.getElementById('latest-version');
        const commitList = document.getElementById('update-commits');

        if (currentEl) currentEl.textContent = updateInfo.current_version || 'Unknown';
        if (latestEl) latestEl.textContent = updateInfo.latest_version || updateInfo.latest_tag || 'Unknown';

        if (commitList) {
            commitList.innerHTML = '';
            const commits = updateInfo.commits || [];
            if (commits.length === 0) {
                const li = document.createElement('li');
                li.textContent = 'No commit details available';
                commitList.appendChild(li);
            } else {
                commits.forEach(commit => {
                    const li = document.createElement('li');
                    li.textContent = commit;
                    commitList.appendChild(li);
                });
            }
        }

        this.setUpdateStatusText(updateInfo.error ? `Last check failed: ${updateInfo.error}` : '', !!updateInfo.error);
    }

    setUpdateStatusText(message, isError = false) {
        const statusEl = document.getElementById('update-status-text');
        if (!statusEl) return;

        statusEl.textContent = message || '';
        statusEl.style.color = isError ? 'var(--danger-color)' : 'var(--secondary-color)';
    }

    handleUpdateStatus(updateInfo) {
        if (!updateInfo) return;
        this.updateStatus = updateInfo;

        if (updateInfo.updating) {
            this.populateUpdateModal(updateInfo);
            this.setUpdateStatusText('Updating application and restarting. Please wait...', false);
            this.showUpdateModal();
            return;
        }

        const latestVersion = updateInfo.latest_version || updateInfo.latest_tag;
        const shouldPrompt =
            updateInfo.update_available &&
            this.isQueuePaused &&
            latestVersion &&
            this.lastUpdatePromptedVersion !== latestVersion;

        if (shouldPrompt) {
            this.lastUpdatePromptedVersion = latestVersion;
            this.populateUpdateModal(updateInfo);
            this.showUpdateModal();
        }
    }

    async manualUpdateCheck() {
        this.setUpdateButtonsDisabled(true);
        this.setUpdateStatusText('Checking for updates...', false);

        try {
            const response = await fetch('/api/update/check', { method: 'POST' });

            if (!response.ok) {
                const errorMessage = await this.extractErrorMessage(response);
                throw new Error(errorMessage || 'Failed to check for updates');
            }

            const data = await response.json();
            this.updateStatus = data;

            const latestVersion = data.latest_version || data.latest_tag;
            if (latestVersion) {
                this.lastUpdatePromptedVersion = latestVersion;
            }

            this.populateUpdateModal(data);

            const hasUpdate = data.update_available;
            const statusMessage = hasUpdate
                ? 'Update available. Review the changes below and apply when ready.'
                : 'Already on the latest version.';

            this.setUpdateStatusText(statusMessage, false);

            if (!hasUpdate) {
                this.showToast('Already on the latest version.', 'info', 4000);
                this.hideUpdateModal();
                return;
            }

            this.showUpdateModal();
        } catch (error) {
            console.error('Failed to check for updates:', error);
            this.showToast('Failed to check for updates: ' + error.message, 'error');
            this.setUpdateStatusText('Failed to check for updates. Please try again.', true);
        } finally {
            const keepDisabled = this.updateStatus && this.updateStatus.updating;
            if (!keepDisabled) {
                this.setUpdateButtonsDisabled(false);
            }
        }
    }

    setUpdateButtonsDisabled(disabled) {
        const applyButton = document.getElementById('apply-update-btn');
        const closeButton = document.getElementById('close-update-btn');
        if (applyButton) applyButton.disabled = disabled;
        if (closeButton) closeButton.disabled = disabled;
    }

    async applyUpdate() {
        this.setUpdateButtonsDisabled(true);
        this.setUpdateStatusText('Applying update...', false);

        try {
            const response = await fetch('/api/update/apply', { method: 'POST' });

            if (!response.ok) {
                const errorMessage = await this.extractErrorMessage(response);
                throw new Error(errorMessage || 'Failed to apply update');
            }

            const data = await response.json();

            if (!this.updateStatus) this.updateStatus = {};
            this.updateStatus.updating = data.status === 'updating';

            if (data.status === 'updating') {
                this.setUpdateStatusText('Update applied. The application will restart shortly.', false);
                this.showToast('Updating application and restarting...', 'info', 7000);
            } else if (data.status === 'up_to_date') {
                this.setUpdateStatusText('Already on the latest version.', false);
                this.showToast('Already up to date.', 'info', 4000);
                this.hideUpdateModal();
            } else if (data.status === 'in_progress') {
                this.setUpdateStatusText('Update already in progress.', false);
            } else {
                this.setUpdateStatusText(data.message || 'Update response received.', false);
            }
        } catch (error) {
            console.error('Failed to apply update:', error);
            this.showToast('Failed to apply update: ' + error.message, 'error');
            this.setUpdateStatusText('Failed to apply update. Please try again.', true);
            this.setUpdateButtonsDisabled(false);
            return;
        }

        const keepDisabled = this.updateStatus && this.updateStatus.updating;
        if (!keepDisabled) {
            this.setUpdateButtonsDisabled(false);
        }
    }

    populateSettingsForm() {
        if (!this.config) return;

        this.initializeEndpointState();

        const form = document.getElementById('settings-form');
        form.default_search_limit.value = this.config.default_search_limit || 50;
        form.max_concurrent_tasks.value = this.config.max_concurrent_tasks || 2;

        // Populate overwrite original setting
        const overwriteOriginal = this.config.overwrite_original !== false; // default to true
        form.overwrite_original.checked = overwriteOriginal;

        this.renderEndpointSelectOptions();
        this.populateEndpointForm();
    }

    useVideoSettings() {
        const endpoint = this.getEndpointById(this.searchEndpointId) || this.endpointConfigs[0];
        if (!endpoint || !endpoint.video_settings) {
            this.showToast('Video settings not available', 'warning');
            return;
        }

        const videoSettings = endpoint.video_settings;
        document.getElementById('max_width').value = videoSettings.width || '';
        document.getElementById('max_height').value = videoSettings.height || '';
        document.getElementById('max_bitrate').value = videoSettings.bitrate || '';
        document.getElementById('max_framerate').value = videoSettings.framerate || '';
        document.getElementById('min_filesize').value = videoSettings.min_filesize || '';
    }

    async saveSettings(formData) {
        try {
            console.log('Saving settings...');

            this.persistEndpointForm();
            if (!this.endpointConfigs || this.endpointConfigs.length === 0) {
                this.endpointConfigs = [this.createEndpointConfig('default')];
                this.settingsEndpointId = this.endpointConfigs[0].id;
            }

            if (!this.settingsEndpointId || !this.getEndpointById(this.settingsEndpointId)) {
                this.settingsEndpointId = this.endpointConfigs[0].id;
            }

            const settings = {
                overwrite_original: formData.get('overwrite_original') === 'on',
                default_search_limit: parseInt(formData.get('default_search_limit')) || 50,
                max_concurrent_tasks: parseInt(formData.get('max_concurrent_tasks')) || 2,
                endpoints: this.endpointConfigs,
                active_endpoint_id: this.settingsEndpointId
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
                this.initializeEndpointState();

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

    async cancelConversion(taskId) {
        try {
            const response = await fetch(`/api/cancel-conversion/${taskId}`, { method: 'POST' });

            if (response.status === 200) {
                const result = await response.json();
                if (result.status === 'cancelled') {
                    this.showToast('Conversion cancelled', 'success');
                    // Force immediate status update to reflect auto-pause if applicable
                    await this.fetchAndUpdateConversionStatus();
                } else if (result.status === 'not_cancellable') {
                    this.showToast('Task cannot be cancelled in its current state', 'warning');
                } else if (result.status === 'already_cancelled') {
                    this.showToast('Task is already cancelled', 'info');
                } else {
                    this.showToast('Task status: ' + result.status, 'info');
                }
                return;
            } else if (response.status === 404) {
                this.showToast('Task not found', 'error');
            }

            // Handle error responses
            const errorData = await response.json().catch(() => ({}));
            const errorMessage = errorData.detail || `HTTP error ${response.status}`;
            this.showToast(`Failed to cancel conversion: ${errorMessage}`, 'error');

        } catch (error) {
            console.error('Failed to cancel conversion:', error);
            this.showToast('Failed to cancel conversion: ' + error.message, 'error');
        }
    }

    async showLog(taskId) {
        try {
            // Fetch the actual log content from the server
            const response = await fetch(`/api/conversion-log/${taskId}`);
            if (response.ok) {
                const logData = await response.json();
                const logContent = logData.log || 'No log content available';
                document.getElementById('log-content').textContent = logContent;
                document.getElementById('log-modal').style.display = 'block';
            } else {
                throw new Error('Failed to fetch log');
            }
        } catch (error) {
            console.error('Failed to load log:', error);
            document.getElementById('log-content').textContent = 'Error loading log: ' + error.message;
            document.getElementById('log-modal').style.display = 'block';
        }
    }

    hideLogModal() {
        document.getElementById('log-modal').style.display = 'none';
        document.getElementById('log-content').textContent = '';
    }

    async retryConversion(taskId) {
        try {
            const response = await fetch(`/api/retry-conversion/${taskId}`, { method: 'POST' });

            if (response.status === 200) {
                const result = await response.json();

                if (result.status === 'retried') {
                    let message = 'Conversion retried. Task is now pending.';
                    if (this.isQueuePaused) {
                        message += ' Queue is paused - start queue to begin processing.';
                    }
                    this.showToast(message, 'success');
                } else {
                    this.showToast('Conversion retry status: ' + result.status, 'info');
                }
                // Force immediate status update
                await this.fetchAndUpdateConversionStatus();
                return;
            }

            // Handle error responses
            const errorData = await response.json().catch(() => ({}));
            const errorMessage = errorData.detail || `HTTP error ${response.status}`;

            this.showToast(`Failed to retry conversion: ${errorMessage}`, 'error');

        } catch (error) {
            console.error('Failed to retry conversion:', error);
            if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
                this.showToast('Network error. Please check your connection and try again.', 'error');
            } else {
                this.showToast('Failed to retry conversion: ' + error.message, 'error');
            }
        }
    }

    async retryStashFix(taskId) {
        try {
            this.showToast('Attempting to fix Stash update...', 'info');

            const response = await fetch(`/api/retry-stash-fix/${taskId}`, { method: 'POST' });

            if (response.status === 200) {
                const result = await response.json();
                if (result.status === 'retrying_stash') {
                    this.showToast('Stash fix started. Check logs for details.', 'info');
                } else {
                    this.showToast('Stash fix status: ' + result.status, 'info');
                }
                // Force immediate status update
                await this.fetchAndUpdateConversionStatus();
                return;
            }

            // Handle error responses
            const errorData = await response.json().catch(() => ({}));
            const errorMessage = errorData.detail || `HTTP error ${response.status}`;

            // Show the error message to user
            this.showToast(`Failed to fix Stash: ${errorMessage}`, 'error');

            // Force refresh to show updated task status (may have been reset to pending)
            await this.fetchAndUpdateConversionStatus();

        } catch (error) {
            console.error('Failed to fix Stash update:', error, error.message);
            // Provide more user-friendly error message
            if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
                this.showToast('Network error. Please check your connection and try again.', 'error');
            } else {
                this.showToast('Failed to fix Stash update: ' + error.message, 'error');
            }
        }
    }

    async removeFromQueue(taskId) {
        try {
            const response = await fetch(`/api/remove-from-queue/${taskId}`, { method: 'POST' });
            if (response.status === 200) {
                 const result = await response.json();
                 if (result.status === 'removed') {
                    let message = 'Task removed from queue';
                    if (result.status_was === 'cancelled') {
                        message += ' (temporary files cleaned up)';
                    }
                    this.showToast(message, 'success');
                }
                // Force immediate status update
                await this.fetchAndUpdateConversionStatus();
                return;
            }

            // Handle error responses
            const errorData = await response.json().catch(() => ({}));
            const errorMessage = errorData.detail || `HTTP error ${response.status}`;
            this.showToast(`Failed to remove task from queue: ${errorMessage}`, 'error');

        } catch (error) {
            console.error('Failed to remove from queue:', error);
            this.showToast('Failed to remove task from queue: ' + error.message, 'error');
        }
    }

    renderResults() {
        const tbody = document.querySelector('#results-table tbody');
        const tableContainer = document.querySelector('.table-container');
        const paginationControls = document.querySelectorAll('.pagination-controls');
        const resultsControls = document.querySelector('.results-controls');

        tbody.innerHTML = '';

        if (!this.currentResults || this.currentResults.length === 0) {
            // Only update DOM if we're actually showing the results section
            if (this.resultsSection.style.display !== 'block') {
                if (tableContainer) tableContainer.style.display = 'none';
                paginationControls.forEach(control => control.style.display = 'none');
                if (resultsControls) resultsControls.style.display = 'none';
                return;
            }

            // Hide entire results section when no results
            if (this.resultsSection) this.resultsSection.style.display = 'none';
            if (tableContainer) tableContainer.style.display = 'none';
            paginationControls.forEach(control => control.style.display = 'none');
            if (resultsControls) resultsControls.style.display = 'none';

            // Show "no results" message in table only (no toast)
            const noResultsRow = document.createElement('tr');
            noResultsRow.innerHTML = `<td colspan="10" style="text-align: center; padding: 2rem; color: var(--secondary-color);">No scenes found matching your search criteria</td>`;
            tbody.appendChild(noResultsRow);
            this.syncPaginationControls();
            return;
        }

        // Show results section when there are results
        // Note: Don't force show results section here - let section navigation handle it
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

        const endpointForResults = this.getEndpointById(this.currentResultsEndpointId) || this.endpointConfigs[0];
        const endpointIdForResults = endpointForResults?.id || this.currentResultsEndpointId || this.searchEndpointId;
        const stashBaseUrl = endpointForResults?.stash_url || '';

        pageResults.forEach(scene => {
            const file = scene.files && scene.files.length > 0 ? scene.files[0] : null;
            if (!file) return;

            const isQueued = this.queuedSceneIds.has(this.getQueueSceneKey(scene.id, endpointIdForResults));
            const isSelected = this.selectedScenes.has(scene.id) && !isQueued;
            const checkboxDisabled = isQueued;
            const checkboxTitle = isQueued ? 'Already in conversion queue' : isSelected ? 'Selected for conversion' : 'Click to select';
            const stashSceneUrl = stashBaseUrl ? `${stashBaseUrl}/scenes/${scene.id}` : '#';

            const row = document.createElement('tr');

            row.innerHTML = `
                <td>
                    <input type="checkbox" class="scene-checkbox" value="${scene.id}"
                           ${isSelected ? 'checked' : ''}
                           ${checkboxDisabled ? 'disabled' : ''}
                           title="${checkboxTitle}">
                </td>
                <td class="title-cell" title="${scene.title || 'Untitled'}">
                    <a href="${stashSceneUrl}" target="_blank">${scene.title || 'Untitled'}</a>
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

            // Update tooltip for queued scenes
            if (isQueued) {
                row.querySelector('.scene-checkbox').style.cursor = 'not-allowed';
            }
        });

        this.syncPaginationControls();
        this.updateSelectionControls();
    }

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

            const endpointId = this.searchEndpointId || this.config?.active_endpoint_id || this.endpointConfigs[0]?.id;
            if (endpointId) {
                searchParams.endpoint_id = endpointId;
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
            this.currentResultsEndpointId = endpointId;
            this.currentPage = 1; // Reset to first page on new search
            this.selectedScenes.clear();
            this.syncPaginationControls();
            this.renderResults();

            // Show appropriate message based on results
            if (this.currentResults.length === 0) {
                this.showToast('No scenes found matching your search criteria', 'info');
            } else {
                this.showToast(`Found ${this.currentResults.length} scenes`, 'success');
                // Ensure we're in search section when showing results
                this.showSearchSection();
            }
        } catch (error) {
            console.error('Search failed:', error);
            this.showToast('Search failed: ' + error.message, 'error');
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
        // Get all scene IDs on current page (including queued ones for reference)
        const allCurrentPageSceneIds = this.getCurrentPageSceneIds();
        const endpointId = this.currentResultsEndpointId || this.searchEndpointId;

        // Filter out queued scenes and only select non-queued ones
        const selectableSceneIds = allCurrentPageSceneIds.filter(id =>
            !this.queuedSceneIds.has(this.getQueueSceneKey(id, endpointId))
        );

        // Add all selectable scenes to selected set
        selectableSceneIds.forEach(id => this.selectedScenes.add(id));

        // Re-render to update checkboxes
        this.renderResults();
    }

    selectNone() {
        // Clear all selections (including queued ones won't be selectable anyway)
        this.selectedScenes.clear();
        this.renderResults();
    }

    selectInvert() {
        const currentPageSceneIds = this.getCurrentPageSceneIds();
        const endpointId = this.currentResultsEndpointId || this.searchEndpointId;

        currentPageSceneIds.forEach(id => {
            // Skip queued scenes - they can't be selected
            if (this.queuedSceneIds.has(this.getQueueSceneKey(id, endpointId))) {
                return;
            }

            if (this.selectedScenes.has(id)) {
                this.selectedScenes.delete(id);
            } else {
                this.selectedScenes.add(id);
            }
        });

        this.renderResults();
    }

    getCurrentPageSceneIds() {
        // Start with current results
        const displayResults = [...this.currentResults];

        // Apply sorting if active
        if (this.sortField && this.sortDirection) {
            displayResults.sort((sceneA, sceneB) => {
                const aVal = this.getSortValue(sceneA, this.sortField);
                const bVal = this.getSortValue(sceneB, this.sortField);

                if (typeof aVal === 'string') {
                    const aLower = aVal.toLowerCase();
                    const bLower = bVal.toLowerCase();
                    if (this.sortDirection === 'asc') {
                        return aLower.localeCompare(bLower);
                    } else {
                        return bLower.localeCompare(aLower);
                    }
                } else {
                    return this.sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
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
        const endpointId = this.currentResultsEndpointId || this.searchEndpointId;

        // Count only selectable scenes (non-queued) that are selected
        const selectableSceneIds = currentPageScenes.filter(id => !this.queuedSceneIds.has(this.getQueueSceneKey(id, endpointId)));
        const selectedCount = selectableSceneIds.filter(id => this.selectedScenes.has(id)).length;
        const allSelected = selectedCount === currentPageScenes.length;
        const allSelectableSelected = selectedCount === selectableSceneIds.length && selectableSceneIds.length > 0;

        document.getElementById('select-all-checkbox').checked = allSelected;
        document.getElementById('select-all-checkbox').indeterminate = selectedCount > 0 && !allSelectableSelected;

        // Update the select all checkbox title for clarity
        const selectAllCheckbox = document.getElementById('select-all-checkbox');
        if (selectableSceneIds.length === 0) {
            selectAllCheckbox.title = "No selectable scenes on this page";
            selectAllCheckbox.disabled = true;
        } else {
            selectAllCheckbox.title = `Select all ${selectableSceneIds.length} selectable scenes on this page`;
            selectAllCheckbox.disabled = false;
        }
    }

    async queueConversion() {
        if (this.isFirstRun) {
            this.showToast('Please complete the first-time setup by saving the configuration.', 'warning');
            return;
        }

        // Get only selectable scenes (non-queued) that are selected
        const endpointId = this.currentResultsEndpointId || this.searchEndpointId || this.config?.active_endpoint_id;
        const selectableSelectedScenes = Array.from(this.selectedScenes).filter(
            id => !this.queuedSceneIds.has(this.getQueueSceneKey(id, endpointId))
        );

        if (selectableSelectedScenes.length === 0) {
            this.showToast('Please select at least one scene to convert. Note: Already queued scenes cannot be selected.', 'warning');
            return;
        }

        if (this.selectedScenes.size === 0) {
            this.showToast('Please select at least one scene to convert.', 'warning');
            return;
        }

        const maxAttempts = 5;
        const backoffTimes = [2000, 4000, 8000, 12000];
        const sceneIds = Array.from(selectableSelectedScenes);
        let lastError = '';

        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                const response = await fetch('/api/queue-conversion', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        scene_ids: sceneIds,
                        endpoint_id: endpointId
                    })
                });

                if (response.ok) {
                    this.showConversionSection();
                    this.startSSE();
                    const responseData = await response.json();
                    this.updateQueuedSceneIds(responseData.queue || []);

                    // Start processing if not paused
                    if (!this.isQueuePaused) this.startQueueProcessing();
                    this.showToast(`Queued ${sceneIds.length} scenes for conversion.`, 'success');
                    return;
                }

                const errorMessage = await this.extractErrorMessage(response);
                throw new Error(errorMessage || 'Failed to queue conversion');
            } catch (error) {
                lastError = error.message || 'Unknown error';

                if (attempt < maxAttempts) {
                    const waitTime = backoffTimes[attempt - 1] || backoffTimes[backoffTimes.length - 1];
                    this.showToast(
                        `Queue request failed (attempt ${attempt} of ${maxAttempts}): ${lastError}. Retrying in ${waitTime / 1000} seconds...`,
                        'warning',
                        waitTime + 1000
                    );
                    console.warn(`Queue request failed, retrying in ${waitTime}ms:`, error);
                    await new Promise(resolve => setTimeout(resolve, waitTime));
                } else {
                    console.error('Add to queue failed after retries:', error);
                    this.showToast('Failed to queue conversion: ' + lastError, 'error');
                }
            }
        }
    }

    async toggleQueuePause() {
        try {
            const response = await fetch('/api/toggle-pause', { method: 'POST' });
            if (response.ok) {
                const result = await response.json();
                console.log('toggle-pause response data:', result);
                this.isQueuePaused = result.paused;
                this.updatePauseButton();
                this.showToast(`Queue ${this.isQueuePaused ? 'paused' : 'started'}`, 'info');

                // Force immediate SSE update when unpausing
                if (!this.isQueuePaused) {
                    console.log('Queue started, forcing immediate update');
                    // Force a manual fetch and start processing
                    await this.fetchAndUpdateConversionStatus();
                    this.startQueueProcessing();
                }
            } else {
                throw new Error('Failed to toggle pause');
            }
        } catch (error) {
            console.error('Failed to toggle pause:', error);
            this.showToast('Failed to toggle pause: ' + error.message, 'error');
        }
    }

    updatePauseButton() {
        const btn = document.getElementById('toggle-pause');
        if (btn) {
            btn.textContent = this.isQueuePaused ? 'Start Queued Tasks' : 'Pause Task Queue';
            btn.className = `btn ${this.isQueuePaused ? 'btn-primary' : 'btn-secondary'}`;
        }
    }

    async startQueueProcessing() {
        try {
            const response = await fetch('/api/start-processing', { method: 'POST' });
            if (!response.ok) {
                throw new Error('Failed to start queue processing');
            }
            console.log('Queue processing started');

            // Force immediate status check
            await this.fetchAndUpdateConversionStatus();
        } catch (error) {
            console.error('Failed to start queue processing:', error);
        }
    }

    showConversionSection() {
        console.log('Showing conversion section');

        // Check if we have any tasks to show
        if (!this.lastConversionStatus || !this.lastConversionStatus.queue || this.lastConversionStatus.queue.length === 0) {
            console.log('No tasks in queue, staying in search section');
            this.showToast('Conversion queue is empty', 'info');
            return;
        }

        // Hide other sections
        if (this.searchSection) this.searchSection.style.display = 'none';
        if (this.resultsSection) this.resultsSection.style.display = 'none';

        // Show conversion section
        if (this.conversionSection) this.conversionSection.style.display = 'block';

        // Update navigation buttons - hide "View Conversion Queue" when we're on conversion page
        this.showSearchBtn.style.display = 'inline-block';
        this.showConversionBtn.style.display = 'none';

        // Update UI with cached data immediately
        if (this.lastConversionStatus) {
            console.log('Using cached conversion status:', this.lastConversionStatus);
            this.updateConversionStatus(this.lastConversionStatus);
        } else {
            // If no cached data, fetch current status
            console.log('No cached data, fetching current conversion status');
            this.fetchAndUpdateConversionStatus();
        }

        this.updatePauseButton();

        // Start SSE when showing conversion section
        // Start SSE without blocking UI
        this.startSSE().catch(error => {
            console.error('Failed to start SSE:', error);
        });
    }

    async fetchAndUpdateConversionStatus() {
        try {
            const response = await fetch('/api/conversion-status');
            const statusData = await response.json();
            this.updateConversionStatus(statusData);
        } catch (error) {
            console.error('Failed to fetch conversion status:', error);
            this.showToast('Failed to load conversion queue', 'error');
        }
    }

    showSearchSection() {
        // Hide other sections
        if (this.conversionSection) this.conversionSection.style.display = 'none';

        // Show search sections
        if (this.searchSection) this.searchSection.style.display = 'block';

        // Show results section only if we have results
        if (this.resultsSection && this.currentResults && this.currentResults.length > 0) {
            this.resultsSection.style.display = 'block';
        }

        // Update navigation buttons - show "View Conversion Queue" when we're on search page
        this.showSearchBtn.style.display = 'none';
        if (this.lastConversionStatus && this.lastConversionStatus.queue && this.lastConversionStatus.queue.length > 0) {
            this.showConversionBtn.style.display = 'inline-block';
        }

        // Pause SSE when not viewing conversion section to reduce load
        this.pauseSSE();
    }

    async checkSSEConnection() {
        try {
            const response = await fetch('/api/debug/sse-status');
            const status = await response.json();
            console.log('SSE Debug Status:', status);

            // Log current task statuses
            if (this.lastConversionStatus) {
                console.log('Last conversion status:', this.lastConversionStatus);
            }
        } catch (error) {
            console.error('Failed to check SSE status:', error);
        }
    }

    // Add a periodic check (optional, for debugging)
    // setInterval(() => this.checkSSEConnection(), 10000); // Every 10 seconds

    // Update the startSSE method to add more logging
    async startSSE() {
        // Close existing connection if any
        if (this.eventSource) {
            console.log('Closing existing SSE connection');
            this.eventSource.close();
            this.eventSource = null;
        }

        // Don't start if page is hidden
        if (document.hidden) {
            console.log('Page is hidden, deferring SSE start');
            return;
        }

        console.log('Starting SSE connection');

        // Ensure we have current data before starting SSE
        await this.ensureInitialConversionStatus();

        this.eventSource = new EventSource('/sse');
        console.log('EventSource created, readyState:', this.eventSource.readyState);

        let lastDataHash = null;
        let messageCount = 0;

        this.eventSource.onmessage = (event) => {
            messageCount++;
            const data = JSON.parse(event.data);

            // Create a simple hash to detect actual changes
            const dataHash = JSON.stringify(data.queue) + '|' + data.paused;

            // Only update if data has actually changed
            if (dataHash !== lastDataHash) {
                lastDataHash = dataHash;
                /*
                console.log(`SSE message #${messageCount}: Data changed, updating UI`);
                console.log('SSE Data received:', {
                    queue_length: data.queue.length,
                    active_tasks: data.active.length,
                    paused: data.paused,
                    tasks: data.queue.map(t => ({
                        id: t.task_id,
                        status: t.status,
                        progress: t.progress
                    }))
                });
                */
                this.updateConversionStatus(data);
            } else {
                // console.log(`SSE message #${messageCount}: No change detected`);
            }
        };

        this.eventSource.onopen = () => {
            console.log('SSE connection opened');
        };

        this.eventSource.onerror = (error) => {
            console.error('SSE error:', error, 'readyState:', this.eventSource.readyState);

            if (this.eventSource.readyState === EventSource.CLOSED) {
                console.log('SSE connection closed normally');
            } else {
                console.error('SSE error, attempting to reconnect in 5 seconds');
                if (!document.hidden) {
                    setTimeout(() => this.startSSE(), 5000);
                }
            }
        };
    }

    async ensureInitialConversionStatus() {

        if (!this.lastConversionStatus) {
            try {
                console.log('Fetching initial conversion status before starting SSE');
                const response = await fetch('/api/conversion-status');
                const statusData = await response.json();
                this.lastConversionStatus = statusData;
                // Update UI with initial data
                if (this.conversionSection && this.conversionSection.style.display === 'block') {
                    this.updateConversionStatus(statusData);
                }
            } catch (error) {
                console.error('Failed to fetch initial conversion status:', error);
            }
        }
    }

    updateConversionStatus(statusData) {
        // Store the latest status data
        this.lastConversionStatus = statusData;

        // Only update if we have valid data
        if (statusData && statusData.queue !== undefined) {
            const orderedQueue = this.orderQueue(statusData.queue);
            this.isQueuePaused = statusData.paused !== undefined ? statusData.paused : true;
            this.renderConversionTable(orderedQueue);
            this.updateQueuedSceneIds(orderedQueue);
            this.updateProgressOverview(orderedQueue, statusData.active);
            this.updateConversionUI(orderedQueue);
        }

        if (statusData && statusData.update) {
            this.handleUpdateStatus(statusData.update);
        }
    }

    orderQueue(queue) {
        if (!queue) return [];

        const statusPriority = {
            'processing': 0,
            'error': 1,
            'pending': 2,
            'cancelled': 3,
            'completed_with_warning': 4,
            'completed': 5,
        };

        return [...queue]
            .map((task, index) => ({ task, index }))
            .sort((a, b) => {
                const priorityA = statusPriority[a.task.status] ?? 3;
                const priorityB = statusPriority[b.task.status] ?? 3;

                if (priorityA !== priorityB) {
                    return priorityA - priorityB;
                }

                return a.index - b.index;
            })
            .map(({ task }) => task);
    }

    updateButtonStates(queue) {
        const hasActiveOrPending = queue.some(task =>
            task.status === 'processing' || task.status === 'pending'
        );
        const hasCompleted = queue.some(task => task.status === 'completed' || task.status === 'completed_with_warning');
        const hasErrors = queue.some(task => task.status === 'error');
        const hasProcessing = queue.some(task => task.status === 'processing');
        const hasPending = queue.some(task => task.status === 'pending');

        // Update Cancel All button
        const cancelAllBtn = document.getElementById('cancel-all');
        if (cancelAllBtn) {
            cancelAllBtn.style.display = hasProcessing ? 'inline-block' : 'none';
        }

        // Update Clear Completed button
        const clearCompletedBtn = document.getElementById('clear-completed');
        if (clearCompletedBtn) {
            clearCompletedBtn.style.display = hasCompleted ? 'inline-block' : 'none';
        }

        // Update Clear Errors button
        const clearErrorsBtn = document.getElementById('clear-errors');
        if (clearErrorsBtn) {
            clearErrorsBtn.style.display = hasErrors ? 'inline-block' : 'none';
        }

        // Update Retry All Errors button
        const retryAllErrorsBtn = document.getElementById('retry-all-errors');
        if (retryAllErrorsBtn) {
            retryAllErrorsBtn.style.display = hasErrors ? 'inline-block' : 'none';
        }

        // Update Remove All Pending button
        const removeAllPendingBtn = document.getElementById('remove-all-pending');
        if (removeAllPendingBtn) {
           removeAllPendingBtn.style.display = hasPending ? 'inline-block' : 'none';
        }

        // Update pause button state
        this.updatePauseButton();
        const togglePauseBtn = document.getElementById('toggle-pause');
        if (togglePauseBtn) {
            togglePauseBtn.disabled = !hasActiveOrPending;
        }

        return {
            hasActiveOrPending, hasCompleted, hasErrors,
            hasProcessing, hasPending,
            hasAnyTasks: queue.length > 0
        };
    }

    updateConversionUI(queue) {
        const buttonStates = this.updateButtonStates(queue);
        const hasAnyTasks = buttonStates.hasAnyTasks;

        // Show conversion button if there are tasks (only when we're on search page)
        if (hasAnyTasks && this.conversionSection && this.conversionSection.style.display === 'none') {
            this.showConversionBtn.style.display = 'inline-block';
        }

        // Hide conversion controls and progress overview when there are no active/pending tasks
        if (this.conversionControls) {
            this.conversionControls.style.display = hasAnyTasks ? 'flex' : 'none';
        }
        if (this.progressOverview) {
            this.progressOverview.style.display = hasAnyTasks ? 'block' : 'none';
        }

        // NEW: If there are no tasks and we're in the conversion section, automatically switch back to search
        if (!hasAnyTasks && this.conversionSection && this.conversionSection.style.display === 'block') {
            console.log('Queue is empty, automatically switching back to search section');
            this.showSearchSection();
        }

        return buttonStates;
    }

    async cancelAllConversions() {
        try {
            const response = await fetch('/api/cancel-all-conversions', { method: 'POST' });
            if (response.ok) {
                const data = await response.json();
                this.showToast(`Cancelled ${data.count || 0} conversion${(data.count || 0) === 1 ? '' : 's'}`, 'info');
                await this.fetchAndUpdateConversionStatus();
            } else {
                throw new Error('Failed to cancel all conversions');
            }
        } catch (error) {
            console.error('Failed to cancel all conversions:', error);
            this.showToast('Failed to cancel all conversions: ' + error.message, 'error');
        }
    }

    async clearCompleted() {
        try {
            const response = await fetch('/api/clear-completed', { method: 'POST' });
            if (response.ok) {
                this.showToast('Cleared completed tasks', 'success');
                await this.fetchAndUpdateConversionStatus();
            } else {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || 'Failed to clear completed tasks');
            }
        } catch (error) {
            console.error('Failed to clear completed tasks:', error);
            this.showToast('Failed to clear completed tasks: ' + error.message, 'error');
        }
    }

    async clearErrors() {
        try {
            const response = await fetch('/api/clear-errors', { method: 'POST' });
            if (response.ok) {
                this.showToast('Cleared errored tasks', 'success');
                await this.fetchAndUpdateConversionStatus();
            } else {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || 'Failed to clear errored tasks');
            }
        } catch (error) {
            console.error('Failed to clear errored tasks:', error);
            this.showToast('Failed to clear errored tasks: ' + error.message, 'error');
        }
    }

    async retryAllErrors() {
        try {
            const response = await fetch('/api/retry-all-errors', { method: 'POST' });
            if (response.ok) {
                const data = await response.json();
                this.showToast(`Retrying ${data.count || 0} errored task${(data.count || 0) === 1 ? '' : 's'}`, 'info');
                await this.fetchAndUpdateConversionStatus();
            } else {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || 'Failed to retry errored tasks');
            }
        } catch (error) {
            console.error('Failed to retry errored tasks:', error);
            this.showToast('Failed to retry errored tasks: ' + error.message, 'error');
        }
    }

    renderConversionTable(queue) {
        const tbody = document.querySelector('#conversion-table tbody');
        const tableContainer = document.querySelector('.conversion-section .table-container');
        const isMobile = window.matchMedia('(max-width: 768px)').matches;
        tbody.innerHTML = '';
        const statusDisplay = {
            'pending': { text: 'pending', icon: '&#9203;' },
            'processing': { text: 'processing', icon: '&#9201;' },
            'completed': { text: 'completed', icon: '&#9989;' },
            'completed_with_warning': { text: 'warning', icon: '&#9888;' },
            'error': { text: 'error', icon: '&#10060;' },
            'cancelled': { text: 'cancelled', icon: '&#128683;' }
        };

        const getStatusContent = (status) => {
            const display = statusDisplay[status] || { text: status, icon: '&#8505;' };

            if (isMobile) {
                return `
                    <span class="status-icon" aria-label="${display.text}" title="${display.text}">${display.icon}</span>
                    <span class="sr-only">${display.text}</span>
                `;
            }

            return display.text;
        };

        if (!queue || queue.length === 0) {
            // Hide table when no queue items
            if (tableContainer) tableContainer.style.display = 'none';
            // Show empty message
            const emptyRow = document.createElement('tr');
            emptyRow.innerHTML = `<td colspan="4" style="text-align: center; padding: 2rem; color: var(--secondary-color);">No conversion tasks in queue</td>`;
            tbody.appendChild(emptyRow);
            return;
        }

        // Show table when there are queue items
        if (tableContainer) tableContainer.style.display = 'block';

        queue.forEach(task => {
            const sceneTitle = task.scene.title || 'Untitled';
            const endpoint = this.getEndpointById(task.endpoint_id) || this.endpointConfigs[0];
            const stashSceneUrl = endpoint?.stash_url ? `${endpoint.stash_url}/scenes/${task.scene.id}` : '#';

            // Determine task status
            const isError = task.status === 'error';
            const isCancelled = task.status === 'cancelled';
            const isWarning = task.status === 'completed_with_warning';
            const isPending = task.status === 'pending';
            const isProcessing = task.status === 'processing';
            const isCompleted = task.status === 'completed';

            const hasErrorDetail = task.error && task.error.length > 0;
            const fileDetails = task.scene.files && task.scene.files.length > 0 ? task.scene.files[0] : null;
            const fileName = fileDetails?.basename || 'Unknown file';
            const filePath = fileDetails?.path || 'Unknown file';

            // Determine what to display in progress column
            let progressDisplay = '';
            if (isProcessing || isPending || isCompleted) {
                // Show progress bar for active/in-progress tasks
                progressDisplay = `
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${task.progress}%"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.875rem; margin-top: 0.25rem;">
                        <span>${task.progress.toFixed(1)}%</span>
                        ${isProcessing && task.eta && task.eta > 0 ? `<span style="color: var(--secondary-color);">ETA: ${this.formatTime(task.eta)}</span>` : ''}
                    </div>`;
            } else {
                // For error/cancelled/warning, just show status text
                progressDisplay = `<div style="padding: 0.5rem; color: var(--secondary-color);">${task.error || 'No progress available'}</div>`;
            }

            // Determine which buttons to show based on status
            let actionButtons = '';

            if (isError || isCancelled) {
                actionButtons = `<button class="btn btn-secondary btn-sm" data-task-id="${task.task_id}" data-action="show-log" title="View conversion log">Log</button>
                                 <button class="btn btn-primary btn-sm" data-task-id="${task.task_id}" data-action="retry" title="Retry this conversion">Retry</button>`;
            } else if (isWarning) {
                actionButtons = `<button class="btn btn-secondary btn-sm" data-task-id="${task.task_id}" data-action="show-log" title="View conversion log">Log</button>
                                 <button class="btn btn-warning btn-sm" data-task-id="${task.task_id}" data-action="retry-stash" title="Retry only the Stash update">Fix Stash</button>`;
            } else if (task.status === 'processing') {
                actionButtons = `<button class="btn btn-danger btn-sm" data-task-id="${task.task_id}" data-action="cancel" title="Cancel this conversion">Cancel</button>`;
            } else if (isPending || isCompleted || isCancelled) {
                actionButtons = `<button class="btn btn-secondary btn-sm" data-task-id="${task.task_id}" data-action="remove" title="Remove from queue">Remove</button>`;
            }

            if (isMobile) {
                const primaryRow = document.createElement('tr');
                const secondaryRow = document.createElement('tr');
                primaryRow.classList.add('conversion-task-row-primary', 'conversion-task-row');
                secondaryRow.classList.add('conversion-task-row-secondary', 'conversion-task-row');

                primaryRow.innerHTML = `
                    <td class="conversion-title" colspan="4" title="${sceneTitle}">
                        <a class="title-link" href="${stashSceneUrl}" target="_blank" rel="noopener noreferrer">${sceneTitle}</a>
                        <div class="conversion-filepath" title="${filePath}">${this.truncatePath(filePath, 80)}</div>
                    </td>
                `;

                secondaryRow.innerHTML = `
                    <td class="conversion-status status-${task.status}" ${hasErrorDetail ? `title="${task.error}"` : ''}>${getStatusContent(task.status)}</td>
                    <td class="conversion-progress" colspan="2">
                        ${progressDisplay}
                    </td>
                    <td class="conversion-actions">
                        <div class="action-buttons-container">
                        ${actionButtons}
                        </div>
                    </td>
                `;

                const mobileRows = [primaryRow, secondaryRow];

                if (hasErrorDetail) {
                    const statusCell = secondaryRow.querySelector('.conversion-status');
                    statusCell.style.cursor = 'help';
                }

                // Style rows based on status
                if (isError) {
                    mobileRows.forEach(row => row.style.backgroundColor = 'color-mix(in srgb, var(--danger-color) 8%, transparent)');
                }
                if (isCancelled) {
                    mobileRows.forEach(row => row.style.backgroundColor = 'color-mix(in srgb, var(--secondary-color) 12%, transparent)');
                }
                if (isWarning) {
                    mobileRows.forEach(row => row.style.backgroundColor = 'color-mix(in srgb, #ff9800 8%, transparent)');
                }
                if (isCancelled || isError) {
                    mobileRows.forEach(row => row.style.opacity = '0.8');
                }
                if (isPending && hasErrorDetail && task.error.includes('missing')) {
                    mobileRows.forEach(row => row.style.backgroundColor = 'color-mix(in srgb, #ff9800 15%, transparent)');
                }

                tbody.appendChild(primaryRow);
                tbody.appendChild(secondaryRow);
            } else {
                const row = document.createElement('tr');
                row.classList.add('conversion-task-row');

                row.innerHTML = `
                    <td class="conversion-title" title="${sceneTitle}">
                        <a class="title-link" href="${stashSceneUrl}" target="_blank" rel="noopener noreferrer">${sceneTitle}</a>
                        <div class="conversion-filepath" title="${filePath}">${this.truncatePath(filePath, 80)}</div>
                    </td>
                    <td class="conversion-status status-${task.status}" ${hasErrorDetail ? `title="${task.error}"` : ''}>${getStatusContent(task.status)}</td>
                    <td class="conversion-progress">
                        ${progressDisplay}
                    </td>
                    <td class="conversion-actions">
                        <div class="action-buttons-container">
                        ${actionButtons}
                        </div>
                    </td>
                `;

                if (hasErrorDetail) {
                    const statusCell = row.querySelector('.conversion-status');
                    statusCell.style.cursor = 'help';
                }

                if (isError) {
                    row.style.backgroundColor = 'color-mix(in srgb, var(--danger-color) 8%, transparent)';
                }
                if (isCancelled) {
                    row.style.backgroundColor = 'color-mix(in srgb, var(--secondary-color) 12%, transparent)';
                }
                if (isWarning) {
                    row.style.backgroundColor = 'color-mix(in srgb, #ff9800 8%, transparent)';
                }
                if (isCancelled || isError) {
                    row.style.opacity = '0.8';
                }
                if (isPending && hasErrorDetail && task.error.includes('missing')) {
                    row.style.backgroundColor = 'color-mix(in srgb, #ff9800 15%, transparent)';
                }

                tbody.appendChild(row);
            }
        });
    }

    updateProgressOverview(queue, activeTasks) {
        const total = queue.length;
        const completed = queue.filter(task => task.status === 'completed' || task.status === 'error').length;
        const remaining = total - completed;
        const progress = total > 0 ? (completed / total) * 100 : 0;
        const hasActiveOrPending = queue.some(task => task.status === 'processing' || task.status === 'pending');

        document.getElementById('overall-progress').style.width = `${progress}%`;
        document.getElementById('progress-text').innerHTML = `
            <strong>Total Queue Progress</strong><br>
            ${progress.toFixed(1)}% Complete (${completed}/${total} files, ${remaining} remaining)
        `;

        const hasPendingOrProcessing = queue.some(task => task.status === 'processing' || task.status === 'pending');

        if (hasPendingOrProcessing) {
            const etaElement = document.getElementById('eta-text');
            const getTaskDuration = (task) => {
                const file = task?.scene?.files?.[0];
                return file?.duration || 0;
            };

            const activeProcessingTasks = queue.filter(task => task.status === 'processing');
            const currentTaskEta = activeProcessingTasks.length > 0
                ? Math.max(...activeProcessingTasks.map(task => task.eta || 0))
                : null;

            const totalRemainingDuration = queue.reduce((sum, task) => {
                if (task.status !== 'processing' && task.status !== 'pending') return sum;
                const duration = getTaskDuration(task);
                if (!duration) return sum;

                if (task.status === 'processing') {
                    const progress = task.progress || 0;
                    const processed = (progress / 100) * duration;
                    return sum + Math.max(duration - processed, 0);
                }

                return sum + duration;
            }, 0);

            const totalSpeed = activeProcessingTasks.reduce((sum, task) => {
                if (task.speed && task.speed > 0) {
                    return sum + task.speed;
                }
                return sum;
            }, 0);

            const totalEtaSeconds = totalRemainingDuration > 0 && totalSpeed > 0
                ? totalRemainingDuration / totalSpeed
                : null;

            const totalEtaText = totalEtaSeconds !== null ? this.formatTime(totalEtaSeconds) : 'Calculating...';
            const currentEtaText = currentTaskEta !== null ? this.formatTime(currentTaskEta) : null;

            etaElement.innerHTML = `<strong>Total ETA:</strong> ${totalEtaText}${currentEtaText ? ` (ETA of current tasks: ${currentEtaText})` : ''}`;
            etaElement.style.display = 'block';
        } else {
            document.getElementById('eta-text').style.display = 'none';
        }
    }

    async removeAllPending() {
        if (confirm('Are you sure you want to remove all pending conversions from the queue?')) {
            try {
                const response = await fetch('/api/remove-all-pending', { method: 'POST' });
                if (response.ok) {
                    this.showToast('All pending conversions removed from queue', 'success');
                    await this.fetchAndUpdateConversionStatus();
                } else {
                    throw new Error('Failed to remove all pending conversions');
                }
            } catch (error) {
                console.error('Failed to remove all pending conversions:', error);
                this.showToast('Failed to remove all pending conversions: ' + error.message, 'error');
            }
        }
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
