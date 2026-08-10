<script lang="ts">
	import { Plus, Trash2 } from '@lucide/svelte';

	import { Input } from '$lib/components/ui/input';
	import Label from '$lib/components/ui/label/label.svelte';
	import { Button } from '$lib/components/ui/button';
	import { presetsStore } from '$lib/stores/presets.svelte';
	import type { PresetCustomKey } from '$lib/types/presets';

	interface Props {
		presetName: string;
	}

	let { presetName }: Props = $props();

	const preset = $derived(presetsStore.getPreset(presetName));
	const customKeys = $derived(preset?.customKeys ?? []);

	let newKey = $state('');
	let newValue = $state('');

	function addCustomKey() {
		if (!newKey.trim()) return;

		const preset = presetsStore.getPreset(presetName);

		if (preset) {
			preset.customKeys.push({ key: newKey.trim(), value: newValue.trim() });
			newKey = '';
			newValue = '';
		}
	}

	function updateCustomKey(index: number, field: 'key' | 'value', value: string) {
		const preset = presetsStore.getPreset(presetName);

		if (preset?.customKeys[index]) {
			preset.customKeys[index][field] = value;
		}
	}

	function removeCustomKey(index: number) {
		const preset = presetsStore.getPreset(presetName);

		if (preset) {
			preset.customKeys.splice(index, 1);
		}
	}

	function handleAddKey(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			addCustomKey();
		}
	}

	function canAdd(): boolean {
		return newKey.trim().length > 0;
	}

	const hasKeys = $derived(customKeys.length > 0);
</script>

<div class="space-y-4">
	<Label class="text-sm font-medium">Custom Keys</Label>

	{#if hasKeys}
		<div class="space-y-2">
			{#each customKeys as key, index (index)}
				<div class="flex items-center gap-2">
					<Input value={key.key} oninput={(e) => updateCustomKey(index, 'key', e.currentTarget.value)} class="flex-1" disabled />

					<span class="text-muted-foreground">=</span>

					<Input value={key.value} oninput={(e) => updateCustomKey(index, 'value', e.currentTarget.value)} class="flex-1" />

					<button type="button" onclick={() => removeCustomKey(index)} class="inline-flex h-8 w-8 items-center justify-center rounded transition-colors hover:bg-muted">
						<Trash2 class="h-4 w-4 text-muted-foreground" />
					</button>
				</div>

{/each}
		</div>

{/if}

<div class="flex items-center gap-2">
	<Input bind:value={newKey} placeholder="Key" onkeydown={handleAddKey} class="w-32" />

	<span class="text-muted-foreground">=</span>

	<Input bind:value={newValue} placeholder="Value" onkeydown={(e) => { if (e.key === 'Enter') addCustomKey(); }} class="flex-1" />

	<Button size="sm" variant="outline" onclick={addCustomKey} disabled={!canAdd()}>
		<Plus class="h-4 w-4" />

		Add
	</Button>
</div>

<p class="text-xs text-muted-foreground">Add arbitrary INI keys not in the standard catalog.</p>
</div>
