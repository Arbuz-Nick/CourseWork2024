<template>
  <q-page :id="$style.mainPage">
    <q-btn :id="$style.chooseButton" label="get file" @onclick="handle()" />
    <q-file
      :id="$style.filePicker"
      :model-value="model"
      label="Choose audiofile"
      @update:model-value="handle()"
    ></q-file>
    <q-btn
      :id="$style.runButton"
      label="Run denoising"
      @onclick="runDenoising()"
    />
    <audio v-if="denoisingIsDone" :src="getCurrentAudioPath()" />
  </q-page>
</template>

<script setup lang="ts">
import { ref } from 'vue';
const model = ref<File | undefined>(undefined);
const denoisingIsDone = ref<boolean>(false);

// function checkFileType(files: []) {
//   return files.filter((file) => file.type === 'image/png');
// }

function handle() {
  console.warn(model.value);
}

function getCurrentAudioPath(): string {
  return '~/Projects/Study/CourseWork/data/clean_speech/female_84-121123-0000.wav';
}

function runDenoising() {
  console.warn('aaa', model.value);
}

defineOptions({
  name: 'IndexPage',
});
</script>

<style module lang="scss">
#mainPage {
  justify-content: center;
  align-items: center;
  display: flex;
  flex-direction: column;
  gap: 8px;

  #filePicker {
    width: 100%;
  }
  #runButton {
    width: 20%;
  }
  #chooseButton {
    width: 20%;
  }
}
</style>
