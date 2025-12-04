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

            // Load pause state
            this.isQueuePaused = statusData.paused !== undefined ? statusData.paused : true;

            // Store the initial conversion status
            this.lastConversionStatus = statusData;
            this.updateQueuedSceneIds(statusData.queue);
            this.updateConversionUI(statusData.queue);

            // Track queued scene IDs
            this.updateQueuedSceneIds(statusData.queue);

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
        const newIds = new Set(queue ? queue.map(task => task.scene.id) : []);
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

        // CRF slider value display
        const crfSlider = document.getElementById('crf');
        const crfValue = document.getElementById('crf-value');
        if (crfSlider && crfValue) {
            crfSlider.addEventListener('input', (e) => {
                crfValue.textContent = e.target.value;
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
            const selectableSceneIds = currentPageScenes.filter(id => !this.queuedSceneIds.has(id));

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

        // Populate overwrite original setting
        const overwriteOriginal = this.config.overwrite_original !== false; // default to true
        form.overwrite_original.checked = overwriteOriginal;

        const videoSettings = this.config.video_settings || {};
        form.width.value = videoSettings.width || '';
        form.height.value = videoSettings.height || '';
        form.bitrate.value = videoSettings.bitrate || '';
        form.framerate.value = videoSettings.framerate || '';
        form.buffer_size.value = videoSettings.buffer_size || '';
        form.container.value = videoSettings.container || '';
        // Populate CRF setting
        form.crf.value = videoSettings.crf || 26;
        document.getElementById('crf-value').textContent = videoSettings.crf || 26;
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
                overwrite_original: formData.get('overwrite_original') === 'on',
                default_search_limit: parseInt(formData.get('default_search_limit')) || 50,
                max_concurrent_tasks: parseInt(formData.get('max_concurrent_tasks')) || 2,
                path_mappings: formData.get('path_mappings') ? formData.get('path_mappings').split('\n').filter(m => m.trim()) : [],
                video_settings: {
                    width: parseInt(formData.get('width')) || 1280,
                    height: parseInt(formData.get('height')) || 720,
                    bitrate: formData.get('bitrate') || '1000k',
                    framerate: parseFloat(formData.get('framerate')) || 30,
                    buffer_size: formData.get('buffer_size') || '2000k',
                    container: formData.get('container') || 'mp4',
                    crf: parseInt(formData.get('crf')) || 26  // ADDED: CRF value
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

    async cancelConversion(taskId) {
        try {
            const response = await fetch(`/api/cancel-conversion/${taskId}`, { method: 'POST' });

            if (response.status === 200) {
                const result = await response.json();
                if (result.status === 'cancelled') {
                    this.showToast('Conversion cancelled', 'success');
                } else if (result.status === 'not_cancellable') {
                    this.showToast('Task cannot be cancelled in its current state', 'warning');
                } else if (result.status === 'already_cancelled') {
                    this.showToast('Task is already cancelled', 'info');
                } else {
                    this.showToast('Task status: ' + result.status, 'info');
                }
                // Force immediate status update
                await this.fetchAndUpdateConversionStatus();
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
                    this.showToast('Conversion retried. Task is now pending.', 'success');
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

        pageResults.forEach(scene => {
            const file = scene.files && scene.files.length > 0 ? scene.files[0] : null;
            if (!file) return;

            const isQueued = this.queuedSceneIds.has(scene.id);
            const isSelected = this.selectedScenes.has(scene.id) && !isQueued;
            const checkboxDisabled = isQueued;
            const checkboxTitle = isQueued ? 'Already in conversion queue' : isSelected ? 'Selected for conversion' : 'Click to select';

            const row = document.createElement('tr');

            row.innerHTML = `
                <td>
                    <input type="checkbox" class="scene-checkbox" value="${scene.id}"
                           ${isSelected ? 'checked' : ''}
                           ${checkboxDisabled ? 'disabled' : ''}
                           title="${checkboxTitle}">
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

        // Filter out queued scenes and only select non-queued ones
        const selectableSceneIds = allCurrentPageSceneIds.filter(id =>
            !this.queuedSceneIds.has(id)
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

        currentPageSceneIds.forEach(id => {
            // Skip queued scenes - they can't be selected
            if (this.queuedSceneIds.has(id)) {
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

        // Count only selectable scenes (non-queued) that are selected
        const selectableSceneIds = currentPageScenes.filter(id => !this.queuedSceneIds.has(id));
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
        const selectableSelectedScenes = Array.from(this.selectedScenes).filter(id => !this.queuedSceneIds.has(id));

        if (selectableSelectedScenes.length === 0) {
            this.showToast('Please select at least one scene to convert. Note: Already queued scenes cannot be selected.', 'warning');
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
                const responseData = await response.json();
                this.updateQueuedSceneIds(responseData.queue || []);

                // Start processing if not paused
                if (!this.isQueuePaused) this.startQueueProcessing();
                this.showToast(`Queued ${this.selectedScenes.size} scenes for conversion.`, 'success');
            } else {
                throw new Error('Failed to queue conversion');
            }
        } catch (error) {
            console.error('Add to queue failed:', error);
            this.showToast('Failed to queue conversion: ' + error.message, 'error');
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
            this.isQueuePaused = statusData.paused !== undefined ? statusData.paused : true;
            this.renderConversionTable(statusData.queue);
            this.updateQueuedSceneIds(statusData.queue);
            this.updateProgressOverview(statusData.queue, statusData.active);
            this.updateConversionUI(statusData.queue);
        }
    }

    updateButtonStates(queue) {
        const hasActiveOrPending = queue.some(task =>
            task.status === 'processing' || task.status === 'pending'
        );
        const hasCompleted = queue.some(task => task.status === 'completed');
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
            clearCompletedBtn.style.display = (hasCompleted || hasErrors) ? 'inline-block' : 'none';
        }

        // Update Remove All Pending button
        const removeAllPendingBtn = document.getElementById('remove-all-pending');
        if (removeAllPendingBtn) {
           removeAllPendingBtn.style.display = hasPending ? 'inline-block' : 'none';
        }

        // Update pause button state
        this.updatePauseButton();
        document.getElementById('toggle-pause').disabled = !hasActiveOrPending;

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
    }

    renderConversionTable(queue) {
        const tbody = document.querySelector('#conversion-table tbody');
        const tableContainer = document.querySelector('.conversion-section .table-container');
        tbody.innerHTML = '';
        const statusDisplayText = {
            'pending': 'pending',
            'processing': 'processing',
            'completed': 'completed',
            'completed_with_warning': 'warning',
            'error': 'error',
            'cancelled': 'cancelled'
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
            const row = document.createElement('tr');
            const sceneTitle = task.scene.title || 'Untitled';

            // Determine task status
            const isError = task.status === 'error';
            const isCancelled = task.status === 'cancelled';
            const isWarning = task.status === 'completed_with_warning';
            const isPending = task.status === 'pending';
            const isProcessing = task.status === 'processing';
            const isCompleted = task.status === 'completed';

            const hasErrorDetail = task.error && task.error.length > 0;
            const fileName = task.scene.files && task.scene.files.length > 0 ?
                task.scene.files[0].basename : 'Unknown file';

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

            row.innerHTML = `
                <td class="conversion-title">
                    <div><strong>${sceneTitle}</strong></div>
                    <div style="font-size: 0.875rem; color: var(--secondary-color);">${fileName}</div>
                </td>
                <td class="conversion-status status-${task.status}">${statusDisplayText[task.status] || task.status}</td>
                <td class="conversion-progress">
                    ${progressDisplay}
                </td>
                <td class="conversion-actions">
                    <div class="action-buttons-container">
                    ${actionButtons}
                    </div>
                </td>
            `;

            // Add error detail tooltip if available
            if (hasErrorDetail) {
                row.querySelector('.conversion-status').title = task.error;
                row.querySelector('.conversion-status').style.cursor = 'help';
            }

            // Style rows based on status
            if (isError) row.style.backgroundColor = 'color-mix(in srgb, var(--danger-color) 8%, transparent)';
            if (isCancelled) row.style.backgroundColor = 'color-mix(in srgb, var(--secondary-color) 12%, transparent)';
            if (isWarning) row.style.backgroundColor = 'color-mix(in srgb, #ff9800 8%, transparent)';
            if (isCancelled || isError) {
                row.style.opacity = '0.8';
            }
            if (isPending && hasErrorDetail && task.error.includes('missing')) {
                // Highlight pending tasks that were reset due to missing files
                row.style.backgroundColor = 'color-mix(in srgb, #ff9800 15%, transparent)';
            }

            tbody.appendChild(row);
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

        const hasProcessingTasks = queue.some(task => task.status === 'processing');
        if (hasProcessingTasks) {
            const activeProcessingTasks = queue.filter(task => task.status === 'processing' && task.eta);
            if (activeProcessingTasks.length > 0) {
                // Use the maximum ETA among active tasks
                const maxEta = Math.max(...activeProcessingTasks.map(task => task.eta || 0));
                const etaElement = document.getElementById('eta-text');
                etaElement.textContent = `ETA: ${this.formatTime(maxEta)}`;
                etaElement.style.display = 'block';
            } else {
                const etaElement = document.getElementById('eta-text');
                etaElement.textContent = 'ETA: Calculating...';
                etaElement.style.display = 'block';
            }
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
